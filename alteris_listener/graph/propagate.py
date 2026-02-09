"""Graph message passing for triage score adjustment.

Runs after Pass 3 (LLM triage), before Pass 4 (extraction).
Adjusts triage scores using structural signals that single-item
classification can't see. Pure SQL — no LLM calls, runs in seconds.

Four propagation rules:
1. Sender reputation dampening — catch newsletter urgency-bait
2. Thread score propagation — coherence within conversation threads
3. Contact tier boost — slight bump for recent inner-circle messages
4. Recipient dampening — CC'd and mass emails get reduced
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from alteris_listener.graph.store import GraphStore

logger = logging.getLogger(__name__)


@dataclass
class PropagationStats:
    """Track changes from a single propagation rule."""
    rule_name: str
    dampened: int = 0
    promoted: int = 0
    total_delta: float = 0.0  # sum of all score changes (for avg calculation)

    @property
    def total_changed(self) -> int:
        return self.dampened + self.promoted

    @property
    def avg_delta(self) -> float:
        if self.total_changed == 0:
            return 0.0
        return self.total_delta / self.total_changed


def _effective_score(row: dict) -> int:
    """Get the current effective score for a node (adjusted if available)."""
    adj = row.get("triage_adjusted")
    if adj is not None:
        return adj
    return row.get("triage_relevant", 1)


def rule_sender_reputation(store: GraphStore) -> PropagationStats:
    """Rule 1: Dampen outlier scores from senders with low median.

    If a sender has 5+ triaged messages and their median score is low,
    any individual message scoring much higher than the median is likely
    misclassified (e.g., newsletter with urgency-bait subject line).

    Exempt: tier-1/2 senders and the user's own email addresses.
    Self-sent emails are the highest signal (intentional reminders/todos).
    """
    from alteris_listener.graph.entities import USER_EMAILS

    stats = PropagationStats(rule_name="sender_reputation")

    # Get senders with enough triaged messages, excluding tier 1/2
    senders = store.conn.execute("""
        SELECT
            n.sender,
            COUNT(*) as cnt,
            cs.importance_tier
        FROM nodes n
        LEFT JOIN contact_stats cs ON cs.contact_id = 'contact:' || LOWER(n.sender)
        WHERE n.triage_relevant IS NOT NULL
          AND n.sender != ''
          AND n.sender IS NOT NULL
        GROUP BY n.sender
        HAVING cnt >= 5
    """).fetchall()

    for row in senders:
        sender = row["sender"]
        sender_count = row["cnt"]
        tier = row["importance_tier"]

        # Exempt tier-1 and tier-2 senders
        if tier in (1, 2):
            continue

        # Exempt user's own email addresses
        if sender.lower() in {e.lower() for e in USER_EMAILS}:
            continue

        # Compute median score for this sender
        median_row = store.conn.execute("""
            SELECT triage_relevant FROM nodes
            WHERE sender = ? AND triage_relevant IS NOT NULL
            ORDER BY triage_relevant
            LIMIT 1 OFFSET ?
        """, (sender, sender_count // 2)).fetchone()

        if not median_row:
            continue
        median_score = median_row[0]

        # Find outlier nodes: score > median + 4
        threshold = median_score + 4
        cap = min(median_score + 2, 3)  # don't cap below 0.3

        outliers = store.conn.execute("""
            SELECT id, COALESCE(triage_adjusted, triage_relevant) as eff_score
            FROM nodes
            WHERE sender = ?
              AND triage_relevant IS NOT NULL
              AND COALESCE(triage_adjusted, triage_relevant) > ?
        """, (sender, threshold)).fetchall()

        for node in outliers:
            old_score = node["eff_score"]
            new_score = max(cap, 3)  # floor at 0.3

            if new_score < old_score:
                store.conn.execute(
                    "UPDATE nodes SET triage_adjusted = ? WHERE id = ?",
                    (new_score, node["id"]),
                )
                stats.dampened += 1
                stats.total_delta += (old_score - new_score)
                logger.debug(
                    "Rule 1: %s from sender=%s: %d → %d (sender median=%d)",
                    node["id"], sender, old_score, new_score, median_score,
                )

    store.conn.commit()
    return stats


def rule_thread_coherence(store: GraphStore) -> PropagationStats:
    """Rule 2: Propagate scores within conversation threads.

    Nodes in the same thread should have correlated scores:
    - Low outlier in active thread → pull up (preserve context)
    - Stronger pull-up if thread has multiple deep-tier nodes
    - High outlier in quiet thread → pull down (suspicious)
    - Self-sent emails are never dampened (intentional reminders)
    """
    from alteris_listener.graph.entities import USER_EMAILS
    user_emails_lower = {e.lower() for e in USER_EMAILS}

    stats = PropagationStats(rule_name="thread_coherence")

    # Get thread-level aggregates for threads with 2+ triaged nodes
    thread_agg = store.conn.execute("""
        SELECT
            thread_id,
            COUNT(*) as cnt,
            AVG(COALESCE(triage_adjusted, triage_relevant)) as mean_score,
            MAX(COALESCE(triage_adjusted, triage_relevant)) as max_score,
            MIN(COALESCE(triage_adjusted, triage_relevant)) as min_score,
            SUM(CASE WHEN COALESCE(triage_adjusted, triage_relevant) >= 7 THEN 1 ELSE 0 END) as deep_count
        FROM nodes
        WHERE triage_relevant IS NOT NULL
          AND thread_id IS NOT NULL
          AND thread_id != ''
        GROUP BY thread_id
        HAVING cnt >= 2
    """).fetchall()

    for thread in thread_agg:
        thread_id = thread["thread_id"]
        mean_score = thread["mean_score"]
        max_score = thread["max_score"]
        deep_count = thread["deep_count"]

        # Get all nodes in this thread
        nodes = store.conn.execute("""
            SELECT id, sender, COALESCE(triage_adjusted, triage_relevant) as eff_score
            FROM nodes
            WHERE thread_id = ?
              AND triage_relevant IS NOT NULL
        """, (thread_id,)).fetchall()

        for node in nodes:
            old_score = node["eff_score"]
            new_score = old_score
            is_self_sent = (node["sender"] or "").lower() in user_emails_lower

            # Low outlier in active thread — pull up
            if old_score < mean_score - 2 and mean_score >= 3:
                if deep_count >= 2 and old_score < 7:
                    # Strong pull: thread has multiple deep nodes, this is
                    # part of the same decision chain. Pull up to 1 below max.
                    new_score = min(old_score + 4, max_score - 1, 7)
                else:
                    # Standard pull: move toward mean
                    new_score = min(old_score + 2, int(mean_score))

            # High outlier in quiet thread — pull down
            # Never dampen self-sent emails (intentional reminders/todos)
            if (old_score > mean_score + 4 and old_score >= max_score
                    and not is_self_sent):
                new_score = max(old_score - 2, int(mean_score) + 1)

            if new_score != old_score:
                # Clamp to valid range
                new_score = max(1, min(9, new_score))
                store.conn.execute(
                    "UPDATE nodes SET triage_adjusted = ? WHERE id = ?",
                    (new_score, node["id"]),
                )
                if new_score > old_score:
                    stats.promoted += 1
                else:
                    stats.dampened += 1
                stats.total_delta += abs(new_score - old_score)
                logger.debug(
                    "Rule 2: %s in thread=%s: %d → %d (thread mean=%.1f, deep_count=%d)",
                    node["id"], thread_id, old_score, new_score, mean_score, deep_count,
                )

    store.conn.commit()
    return stats


def rule_contact_tier_boost(store: GraphStore) -> PropagationStats:
    """Rule 3: Slight boost for recent messages from inner circle.

    Messages from tier-1 contacts scoring 0.3-0.5 that are less than
    30 days old get a +1 bump. Ensures inner-circle context doesn't
    get lost in the lightweight bucket.
    """
    stats = PropagationStats(rule_name="contact_tier_boost")
    now = int(time.time())
    cutoff_30d = now - (30 * 86400)

    # Find candidates first
    candidates = store.conn.execute("""
        SELECT n.id, COALESCE(n.triage_adjusted, n.triage_relevant) as eff_score
        FROM nodes n
        WHERE n.triage_relevant IS NOT NULL
          AND COALESCE(n.triage_adjusted, n.triage_relevant) BETWEEN 3 AND 4
          AND n.timestamp > ?
          AND n.sender IN (
              SELECT REPLACE(contact_id, 'contact:', '')
              FROM contact_stats
              WHERE importance_tier = 1
          )
    """, (cutoff_30d,)).fetchall()

    for node in candidates:
        old_score = node["eff_score"]
        new_score = min(old_score + 1, 5)
        if new_score > old_score:
            store.conn.execute(
                "UPDATE nodes SET triage_adjusted = ? WHERE id = ?",
                (new_score, node["id"]),
            )
            stats.promoted += 1
            stats.total_delta += 1

    store.conn.commit()
    return stats


def rule_recipient_dampening(store: GraphStore) -> PropagationStats:
    """Rule 4: Dampen CC'd and mass-email items.

    If a node has cc_to edges to the user but no sent_to edge, it's a CC.
    If a node has 5+ recipient edges, it's a mass email.
    Both patterns suggest lower personal relevance.

    Tier-1/2 senders exempt.
    """
    stats = PropagationStats(rule_name="recipient_dampening")

    # Find nodes where user is CC'd but not direct recipient
    # These have cc_to edges but no sent_to edges
    cc_only = store.conn.execute("""
        SELECT DISTINCT n.id,
               COALESCE(n.triage_adjusted, n.triage_relevant) as eff_score,
               cs.importance_tier
        FROM nodes n
        JOIN edges e_cc ON e_cc.src = n.id AND e_cc.edge_type = 'cc_to'
        LEFT JOIN edges e_to ON e_to.src = n.id AND e_to.edge_type = 'sent_to'
        LEFT JOIN contact_stats cs ON cs.contact_id = 'contact:' || LOWER(n.sender)
        WHERE n.triage_relevant IS NOT NULL
          AND e_to.src IS NULL  -- no direct sent_to edge
          AND COALESCE(n.triage_adjusted, n.triage_relevant) >= 5
          AND (cs.importance_tier IS NULL OR cs.importance_tier > 2)
    """).fetchall()

    for node in cc_only:
        old_score = node["eff_score"]
        new_score = max(old_score - 2, 3)
        if new_score < old_score:
            store.conn.execute(
                "UPDATE nodes SET triage_adjusted = ? WHERE id = ?",
                (new_score, node["id"]),
            )
            stats.dampened += 1
            stats.total_delta += (old_score - new_score)

    # Find mass emails: nodes with 5+ outgoing recipient edges
    mass_email = store.conn.execute("""
        SELECT n.id,
               COALESCE(n.triage_adjusted, n.triage_relevant) as eff_score,
               cs.importance_tier,
               COUNT(e.dst) as recipient_count
        FROM nodes n
        JOIN edges e ON e.src = n.id AND e.edge_type IN ('sent_to', 'cc_to')
        LEFT JOIN contact_stats cs ON cs.contact_id = 'contact:' || LOWER(n.sender)
        WHERE n.triage_relevant IS NOT NULL
          AND COALESCE(n.triage_adjusted, n.triage_relevant) >= 5
          AND (cs.importance_tier IS NULL OR cs.importance_tier > 2)
        GROUP BY n.id
        HAVING recipient_count >= 5
    """).fetchall()

    for node in mass_email:
        old_score = node["eff_score"]
        new_score = max(old_score - 2, 3)
        if new_score < old_score:
            store.conn.execute(
                "UPDATE nodes SET triage_adjusted = ? WHERE id = ?",
                (new_score, node["id"]),
            )
            stats.dampened += 1
            stats.total_delta += (old_score - new_score)

    store.conn.commit()
    return stats


def rule_source_type_prior(store: GraphStore) -> PropagationStats:
    """Rule 5: Adjust scores based on source type prior.

    Different source types have different base rates for relevance:
    - Emails: default (most noise lives here, triage handles it)
    - Messages (iMessage/Slack): higher prior IF the user participated
      in the thread (replied). OTPs, verification codes, and one-way
      automated texts stay at their triage score.
    - Meetings (Granola notes): highest prior — curated content, almost
      always actionable. Items in ignore get bumped to lightweight.
    """
    from alteris_listener.graph.entities import USER_EMAILS

    stats = PropagationStats(rule_name="source_type_prior")

    # User identifiers for checking participation
    # Phone number and Apple ID for iMessage, email for Slack
    user_identifiers = {e.lower() for e in USER_EMAILS}
    # Add phone number (iMessage sender format)
    user_identifiers.add("+13472670483")
    user_identifiers.add("3472670483")

    # Messages: only bump threads where user has participated
    # First, find threads where user sent at least one message
    user_threads = store.conn.execute("""
        SELECT DISTINCT thread_id
        FROM nodes
        WHERE node_type = 'message'
          AND thread_id IS NOT NULL
          AND thread_id != ''
          AND LOWER(sender) IN ({})
    """.format(",".join("?" * len(user_identifiers))),
        list(user_identifiers),
    ).fetchall()
    user_thread_ids = {row["thread_id"] for row in user_threads}

    # Bump messages in user-participated threads: ignore → lightweight
    msg_candidates = store.conn.execute("""
        SELECT id, thread_id, COALESCE(triage_adjusted, triage_relevant) as eff_score
        FROM nodes
        WHERE triage_relevant IS NOT NULL
          AND node_type = 'message'
          AND COALESCE(triage_adjusted, triage_relevant) < 3
    """).fetchall()

    for node in msg_candidates:
        # Only boost if user participated in this thread
        if node["thread_id"] and node["thread_id"] in user_thread_ids:
            old_score = node["eff_score"]
            new_score = 3  # minimum lightweight for real conversations
            if new_score > old_score:
                store.conn.execute(
                    "UPDATE nodes SET triage_adjusted = ? WHERE id = ?",
                    (new_score, node["id"]),
                )
                stats.promoted += 1
                stats.total_delta += (new_score - old_score)
        # else: OTPs, verification codes, one-way texts stay at triage score

    # Meetings: bump ignore → lightweight (meetings are curated content)
    mtg_candidates = store.conn.execute("""
        SELECT id, COALESCE(triage_adjusted, triage_relevant) as eff_score
        FROM nodes
        WHERE triage_relevant IS NOT NULL
          AND node_type = 'meeting'
          AND COALESCE(triage_adjusted, triage_relevant) < 5
    """).fetchall()

    for node in mtg_candidates:
        old_score = node["eff_score"]
        new_score = 5  # meetings are at least mid-lightweight
        if new_score > old_score:
            store.conn.execute(
                "UPDATE nodes SET triage_adjusted = ? WHERE id = ?",
                (new_score, node["id"]),
            )
            stats.promoted += 1
            stats.total_delta += (new_score - old_score)

    store.conn.commit()
    return stats


def run_propagation(
    store: GraphStore,
    max_iterations: int = 3,
    convergence_threshold: float = 0.01,
    dry_run: bool = False,
) -> dict:
    """Run all message passing rules iteratively until convergence.

    Args:
        store: GraphStore instance.
        max_iterations: Maximum propagation rounds.
        convergence_threshold: Stop when changes < this fraction of triaged nodes.
        dry_run: If True, roll back all changes after computing stats.

    Returns dict with propagation statistics.
    """
    from rich.console import Console
    console = Console()

    triaged_count = store.conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE triage_relevant IS NOT NULL"
    ).fetchone()[0]

    if triaged_count == 0:
        console.print("[red]No triaged nodes found. Run Pass 3 first.[/red]")
        return {"error": "no_triaged_nodes"}

    console.print(f"[bold]Pass 3.5: Message Passing ({triaged_count} triaged nodes)[/bold]")

    if dry_run:
        console.print("[yellow]DRY RUN — changes will be rolled back[/yellow]")

    # Clear any previous adjusted scores
    store.conn.execute("UPDATE nodes SET triage_adjusted = NULL")
    store.conn.commit()

    # Snapshot pre-propagation tier distribution
    pre_tiers = _get_tier_distribution(store)

    all_iteration_stats = []
    start_time = time.time()

    for iteration in range(max_iterations):
        console.print(f"\n  [bold]Iteration {iteration + 1}[/bold]")

        # Snapshot before this iteration
        iter_start_tiers = _get_tier_distribution(store)

        # Run rules in sequence, showing tier state after each
        rules = [
            ("sender_reputation", rule_sender_reputation),
            ("thread_coherence", rule_thread_coherence),
            ("contact_tier_boost", rule_contact_tier_boost),
            ("recipient_dampening", rule_recipient_dampening),
            ("source_type_prior", rule_source_type_prior),
        ]

        iteration_rules = []
        for rule_name, rule_fn in rules:
            before_rule = _get_tier_distribution(store)
            result = rule_fn(store)
            after_rule = _get_tier_distribution(store)
            iteration_rules.append(result)

            # Format rule output
            direction = ""
            if result.dampened > 0:
                direction += f"↓{result.dampened} dampened"
            if result.promoted > 0:
                if direction:
                    direction += ", "
                direction += f"↑{result.promoted} promoted"
            if not direction:
                direction = "no changes"

            # Show tier deltas for this rule
            tier_delta = ""
            for tier_name in ("ignore", "lightweight", "deep"):
                d = after_rule[tier_name] - before_rule[tier_name]
                if d != 0:
                    sign = "+" if d > 0 else ""
                    tier_delta += f" {tier_name[0].upper()}:{sign}{d}"

            console.print(f"    {result.rule_name:25s}  {direction:40s}{tier_delta}")

        all_iteration_stats.append(iteration_rules)

        total_changed = sum(r.total_changed for r in iteration_rules)

        # Show cumulative tier state after iteration
        iter_end_tiers = _get_tier_distribution(store)
        console.print(f"    {'':25s}  ─────────────────────────────────────────")
        console.print(
            f"    {'Tiers after iteration':25s}  "
            f"I:{iter_end_tiers['ignore']}  L:{iter_end_tiers['lightweight']}  D:{iter_end_tiers['deep']}"
            f"  (Δ from start: "
            f"I:{iter_end_tiers['ignore'] - pre_tiers['ignore']:+d}  "
            f"L:{iter_end_tiers['lightweight'] - pre_tiers['lightweight']:+d}  "
            f"D:{iter_end_tiers['deep'] - pre_tiers['deep']:+d})"
        )

        # Check convergence
        change_rate = total_changed / max(triaged_count, 1)
        console.print(f"    Total changes: {total_changed} ({change_rate:.1%} of triaged)")

        if change_rate < convergence_threshold:
            console.print(f"  [green]Converged at iteration {iteration + 1}[/green]")
            break

    elapsed = time.time() - start_time

    # Post-propagation tier distribution
    post_tiers = _get_tier_distribution(store)

    # Show tier changes
    console.print(f"\n  Tier distribution (before → after):")
    for tier_name in ("ignore", "lightweight", "deep"):
        pre = pre_tiers.get(tier_name, 0)
        post = post_tiers.get(tier_name, 0)
        delta = post - pre
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        if delta != 0:
            console.print(f"    {tier_name:12s}: {pre:>5} → {post:>5}  ({delta_str})")
        else:
            console.print(f"    {tier_name:12s}: {pre:>5} → {post:>5}")

    # Show specific tier transitions
    transitions = store.conn.execute("""
        SELECT
            CASE
                WHEN triage_relevant < 3 THEN 'ignore'
                WHEN triage_relevant < 7 THEN 'lightweight'
                ELSE 'deep'
            END as old_tier,
            CASE
                WHEN triage_adjusted < 3 THEN 'ignore'
                WHEN triage_adjusted < 7 THEN 'lightweight'
                ELSE 'deep'
            END as new_tier,
            COUNT(*) as cnt
        FROM nodes
        WHERE triage_adjusted IS NOT NULL
          AND triage_adjusted != triage_relevant
        GROUP BY old_tier, new_tier
        ORDER BY cnt DESC
    """).fetchall()

    if transitions:
        console.print(f"\n  Tier transitions:")
        for t in transitions:
            if t["old_tier"] != t["new_tier"]:
                console.print(f"    {t['old_tier']:>12s} → {t['new_tier']:<12s}: {t['cnt']}")

    adjusted_count = store.conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE triage_adjusted IS NOT NULL"
    ).fetchone()[0]

    console.print(f"\n  [green]Done in {elapsed:.1f}s. {adjusted_count} nodes adjusted.[/green]")

    if dry_run:
        store.conn.execute("UPDATE nodes SET triage_adjusted = NULL")
        store.conn.commit()
        console.print("[yellow]DRY RUN — all changes rolled back[/yellow]")

    store.set_bootstrap_state("last_pass", "propagation")
    store.set_bootstrap_state("propagation_completed_at", str(int(time.time())))

    return {
        "iterations": len(all_iteration_stats),
        "adjusted_count": adjusted_count,
        "pre_tiers": pre_tiers,
        "post_tiers": post_tiers,
        "elapsed_seconds": elapsed,
    }


def _get_tier_distribution(store: GraphStore) -> dict[str, int]:
    """Get current tier distribution using effective scores."""
    result = {}
    for tier_name, lo, hi in [("ignore", 0, 2), ("lightweight", 3, 6), ("deep", 7, 10)]:
        count = store.conn.execute(
            """SELECT COUNT(*) FROM nodes
               WHERE triage_relevant IS NOT NULL
               AND COALESCE(triage_adjusted, triage_relevant) BETWEEN ? AND ?""",
            (lo, hi),
        ).fetchone()[0]
        result[tier_name] = count
    return result
