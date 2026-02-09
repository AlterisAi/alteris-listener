"""Contact deduplication — merge contacts sharing the same real-world person.

Two dedup strategies:
1. display_name match — group contacts with identical display_name
2. domain pattern merge — automated senders with unique-per-message addresses
   (e.g., Slack no-reply-xxx@slack.com, AT&T att-services.cn.xxx@att-mail.com)

After grouping, the canonical contact gets:
- Highest importance_tier from any alias
- Sum of total_messages across all aliases
- Union of sources
- Best reply_ratio (weighted average)
- All edges redirected to canonical contact_id
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict

from alteris_listener.graph.store import GraphStore

logger = logging.getLogger(__name__)

# Gmail ignores dots before @
GMAIL_DOMAINS = {"gmail.com", "googlemail.com"}

# Automated sender patterns: unique-per-message addresses
# These get collapsed to a single canonical based on domain/pattern
AUTOMATED_SENDER_PATTERNS = [
    # Slack: no-reply-XXXXX@slack.com → slack.com
    (r'^no-reply-[a-z0-9]+@slack\.com$', 'slack-noreply@slack.com'),
    # AT&T: att-services.cn.DIGITS@emailXX.att-mail.com
    (r'^att-services\.cn\.\d+@email.+\.att-mail\.com$', 'att-services@att-mail.com'),
    (r'^att_update\.\d+@email.+\.att-mail\.com$', 'att-update@att-mail.com'),
    # Yelp: reply+UUID@messaging.yelp.com
    (r'^reply\+[a-f0-9]+@messaging\.yelp\.com$', 'reply@messaging.yelp.com'),
    # Credit Karma: mail@mailNN.creditkarma.com
    (r'^mail@mail\d+\.creditkarma\.com$', 'mail@creditkarma.com'),
    # QuantConnect forum: forum+NNNNN.hash@... → research@quantconnect.com
    (r'^forum\+\d+\.[a-f0-9]+.+@icloud\.com$', 'forum@quantconnect.com'),
    (r'^research.+@icloud\.com$', 'research@quantconnect.com'),
    # Constant Contact: various@org.ccsend.com
    (r'^.+@.+\.ccsend\.com$', None),  # None = group by display_name instead
    # Generic: noreply+hash@formstack.com
    (r'^noreply\+[a-f0-9]+@formstack\.com$', 'noreply@formstack.com'),
]


def normalize_gmail(email: str) -> str:
    """Normalize Gmail addresses: remove dots, lowercase, strip +suffix."""
    local, domain = email.split('@', 1)
    if domain.lower() in GMAIL_DOMAINS:
        # Remove dots from local part
        local = local.replace('.', '')
        # Remove +suffix (e.g., user+tag@gmail.com → user@gmail.com)
        local = local.split('+')[0]
    return f"{local.lower()}@{domain.lower()}"


def match_automated_pattern(email: str) -> str | None:
    """Check if email matches an automated sender pattern.
    Returns canonical address or None."""
    email_lower = email.lower()
    for pattern, canonical in AUTOMATED_SENDER_PATTERNS:
        if re.match(pattern, email_lower):
            return canonical
    return None


def build_alias_groups(store: GraphStore) -> dict[str, list[str]]:
    """Build groups of contact_ids that represent the same entity.
    
    Returns: {canonical_id: [list of all contact_ids in group]}
    """
    rows = store.conn.execute(
        "SELECT contact_id, display_name, total_messages, importance_tier, reply_ratio, sources "
        "FROM contact_stats"
    ).fetchall()
    
    # Phase 1: automated sender pattern matching
    # Map pattern-matched emails to a canonical
    pattern_groups: dict[str, list[dict]] = defaultdict(list)
    remaining = []
    
    for row in rows:
        cid = row["contact_id"]
        email = cid.replace("contact:", "")
        
        canonical = match_automated_pattern(email)
        if canonical:
            pattern_groups[canonical].append(dict(row))
        else:
            remaining.append(dict(row))
    
    # Phase 2: display_name matching for real people
    # Group remaining contacts by normalized display_name
    name_groups: dict[str, list[dict]] = defaultdict(list)
    no_name = []
    
    for row in remaining:
        name = (row["display_name"] or "").strip()
        if name:
            name_groups[name.lower()].append(row)
        else:
            no_name.append(row)
    
    # Within name groups, remove suspicious single-message contacts
    # that don't share a domain with any other member (likely spoofed senders)
    for name_lower, members in name_groups.items():
        if len(members) < 2:
            continue
        # Extract domains for each member
        domains = set()
        for m in members:
            email = m["contact_id"].replace("contact:", "").lower()
            if '@' in email:
                domains.add(email.split('@')[1])
        
        # Filter out members with 1 message whose domain appears nowhere else
        filtered = []
        for m in members:
            email = m["contact_id"].replace("contact:", "").lower()
            if '@' in email:
                domain = email.split('@')[1]
                # Count how many other members share this domain
                domain_peers = sum(
                    1 for other in members
                    if other is not m
                    and '@' in other["contact_id"]
                    and other["contact_id"].replace("contact:", "").lower().split('@')[1] == domain
                )
                # Single-message contact with unique domain = likely spoofed
                if m["total_messages"] <= 1 and domain_peers == 0 and len(members) > 2:
                    logger.info(
                        "Skipping likely spoofed contact %s (1 msg, unique domain %s) "
                        "from group '%s'",
                        m["contact_id"], domain, name_lower,
                    )
                    continue
            filtered.append(m)
        name_groups[name_lower] = filtered
    
    # Phase 3: Gmail dot-normalization within name groups
    # Within each name group, also check if any emails are gmail-equivalent
    # For contacts without display_name, check gmail equivalence directly
    gmail_groups: dict[str, list[dict]] = defaultdict(list)
    truly_remaining = []
    
    for row in no_name:
        email = row["contact_id"].replace("contact:", "")
        if '@' in email:
            domain = email.split('@')[1].lower()
            if domain in GMAIL_DOMAINS:
                normalized = normalize_gmail(email)
                gmail_groups[normalized].append(row)
                continue
        truly_remaining.append(row)
    
    # Build final alias groups
    # canonical_id → [all contact_ids]
    alias_groups: dict[str, list[str]] = {}
    
    # From pattern groups
    for canonical_email, members in pattern_groups.items():
        if len(members) < 2:
            continue
        # Pick the one with most messages as canonical
        members.sort(key=lambda r: r["total_messages"], reverse=True)
        canonical = members[0]["contact_id"]
        alias_groups[canonical] = [m["contact_id"] for m in members]
    
    # From name groups
    for name_lower, members in name_groups.items():
        if len(members) < 2:
            continue
        # Pick canonical: prefer real addresses over noreply/automated,
        # then highest tier, then most messages
        def canonical_sort_key(r):
            cid = r["contact_id"].replace("contact:", "").lower()
            # Penalize noreply, automated, and generated addresses
            is_noreply = any(x in cid for x in [
                'noreply', 'no-reply', 'no_reply', 'donotreply',
                'mailer-daemon', 'postmaster',
            ])
            is_generated = bool(re.search(r'[a-f0-9]{8,}', cid))  # hex hash in address
            is_phone = cid.startswith('+') or cid.replace('-', '').isdigit()
            
            # Priority: real email > phone > noreply > generated
            if is_noreply or is_generated:
                addr_priority = 3
            elif is_phone:
                addr_priority = 1  # phones are stable identifiers
            else:
                addr_priority = 0  # real email addresses
            
            return (
                addr_priority,                  # lower = better address
                (r["importance_tier"] or 3),     # lower tier = more important
                -r["total_messages"],            # more messages = better
            )
        
        members.sort(key=canonical_sort_key)
        canonical = members[0]["contact_id"]
        alias_groups[canonical] = [m["contact_id"] for m in members]
    
    # From gmail groups
    for normalized, members in gmail_groups.items():
        if len(members) < 2:
            continue
        members.sort(key=lambda r: r["total_messages"], reverse=True)
        canonical = members[0]["contact_id"]
        alias_groups[canonical] = [m["contact_id"] for m in members]
    
    return alias_groups


def merge_contact_stats(store: GraphStore, canonical_id: str, alias_ids: list[str]) -> dict:
    """Merge multiple contact_stats rows into the canonical one.
    
    Returns dict with merge details.
    """
    others = [cid for cid in alias_ids if cid != canonical_id]
    if not others:
        return {"canonical": canonical_id, "merged": 0}
    
    # Load all stats
    all_stats = []
    for cid in alias_ids:
        row = store.conn.execute(
            "SELECT * FROM contact_stats WHERE contact_id = ?", (cid,)
        ).fetchone()
        if row:
            all_stats.append(dict(row))
    
    if not all_stats:
        return {"canonical": canonical_id, "merged": 0}
    
    # Compute merged values
    best_tier = min(s.get("importance_tier", 3) for s in all_stats)
    total_msgs = sum(s.get("total_messages", 0) for s in all_stats)
    total_sent = sum(s.get("sent_to_count", 0) for s in all_stats)
    total_received = sum(s.get("recv_from_count", 0) for s in all_stats)
    
    # Weighted average reply_ratio
    total_weight = sum(s.get("total_messages", 0) for s in all_stats)
    if total_weight > 0:
        weighted_rr = sum(
            s.get("reply_ratio", 0) * s.get("total_messages", 0)
            for s in all_stats
        ) / total_weight
    else:
        weighted_rr = 0
    
    # Union of sources
    all_sources = set()
    for s in all_stats:
        src = s.get("sources", "[]")
        if isinstance(src, str):
            try:
                all_sources.update(json.loads(src))
            except (json.JSONDecodeError, TypeError):
                pass
    
    # Best display_name (longest non-empty, or first non-empty)
    names = [s.get("display_name", "") for s in all_stats if s.get("display_name")]
    best_name = max(names, key=len) if names else ""
    
    # Update canonical
    store.conn.execute("""
        UPDATE contact_stats SET
            importance_tier = ?,
            total_messages = ?,
            sent_to_count = ?,
            recv_from_count = ?,
            reply_ratio = ?,
            sources = ?,
            display_name = ?
        WHERE contact_id = ?
    """, (
        best_tier, total_msgs, total_sent, total_received,
        weighted_rr, json.dumps(sorted(all_sources)), best_name,
        canonical_id,
    ))
    
    # Redirect edges from aliases to canonical
    for old_id in others:
        # Update edges where old contact is src
        store.conn.execute(
            "UPDATE OR IGNORE edges SET src = ? WHERE src = ?",
            (canonical_id, old_id),
        )
        # Update edges where old contact is dst
        store.conn.execute(
            "UPDATE OR IGNORE edges SET dst = ? WHERE dst = ?",
            (canonical_id, old_id),
        )
        # Delete edges that would create duplicates (OR IGNORE skips them)
        store.conn.execute("DELETE FROM edges WHERE src = ? OR dst = ?", (old_id, old_id))
        
        # Delete the old contact_stats row
        store.conn.execute("DELETE FROM contact_stats WHERE contact_id = ?", (old_id,))
        
        # Delete the old contact node
        store.conn.execute("DELETE FROM nodes WHERE id = ?", (old_id,))
    
    return {"canonical": canonical_id, "merged": len(others), "best_tier": best_tier}


def run_dedup(store: GraphStore, dry_run: bool = False) -> dict:
    """Run contact deduplication.
    
    Args:
        store: GraphStore instance.
        dry_run: If True, show what would happen without making changes.
    
    Returns dict with dedup statistics.
    """
    from rich.console import Console
    from rich.table import Table
    console = Console()
    
    console.print("[bold]Contact Deduplication[/bold]")
    
    # Count before
    before_count = store.conn.execute("SELECT COUNT(*) FROM contact_stats").fetchone()[0]
    console.print(f"  Contacts before: {before_count}")
    
    if dry_run:
        console.print("[yellow]DRY RUN — no changes will be made[/yellow]")
    
    # Build alias groups
    alias_groups = build_alias_groups(store)
    
    total_aliases = sum(len(v) for v in alias_groups.values())
    total_to_merge = total_aliases - len(alias_groups)  # subtract canonicals
    
    console.print(f"  Alias groups found: {len(alias_groups)}")
    console.print(f"  Contacts to merge: {total_to_merge}")
    
    if not alias_groups:
        console.print("[green]No duplicates found.[/green]")
        return {"before": before_count, "after": before_count, "groups": 0}
    
    # Show top groups
    table = Table(title="Top Alias Groups (by member count)")
    table.add_column("Canonical", style="cyan", max_width=40)
    table.add_column("Name", max_width=20)
    table.add_column("Members", justify="right")
    table.add_column("Aliases (sample)", style="dim", max_width=60)
    
    sorted_groups = sorted(alias_groups.items(), key=lambda x: len(x[1]), reverse=True)
    for canonical, members in sorted_groups[:20]:
        name_row = store.conn.execute(
            "SELECT display_name FROM contact_stats WHERE contact_id = ?",
            (canonical,)
        ).fetchone()
        name = (name_row["display_name"] if name_row else "") or ""
        
        sample = [m.replace("contact:", "") for m in members if m != canonical][:3]
        table.add_row(
            canonical.replace("contact:", "")[:40],
            name[:20],
            str(len(members)),
            ", ".join(s[:25] for s in sample),
        )
    
    console.print(table)
    
    if dry_run:
        console.print("[yellow]DRY RUN — no changes made[/yellow]")
        return {
            "before": before_count,
            "after": before_count - total_to_merge,
            "groups": len(alias_groups),
            "dry_run": True,
        }
    
    # Execute merges
    merged_count = 0
    tier_upgrades = 0
    start = time.time()
    
    for canonical, members in alias_groups.items():
        result = merge_contact_stats(store, canonical, members)
        merged_count += result["merged"]
        if result.get("best_tier", 3) < 3:
            tier_upgrades += 1
    
    store.conn.commit()
    
    # Recompute tiers after merge
    store.recompute_contact_tiers()
    
    after_count = store.conn.execute("SELECT COUNT(*) FROM contact_stats").fetchone()[0]
    elapsed = time.time() - start
    
    console.print(f"\n  Contacts after: {after_count} (removed {before_count - after_count})")
    console.print(f"  Tier upgrades: {tier_upgrades}")
    console.print(f"  [green]Done in {elapsed:.1f}s[/green]")
    
    # Show updated tier 1 contacts
    t1 = store.conn.execute(
        "SELECT contact_id, display_name, total_messages, importance_tier "
        "FROM contact_stats WHERE importance_tier = 1 ORDER BY total_messages DESC"
    ).fetchall()
    
    if t1:
        console.print(f"\n  Tier 1 contacts after dedup:")
        for row in t1:
            cid = row["contact_id"].replace("contact:", "")
            name = row["display_name"] or cid
            console.print(f"    {name:30s}  {row['total_messages']:>4d} msgs  ({cid[:40]})")
    
    store.set_bootstrap_state("dedup_completed_at", str(int(time.time())))
    
    return {
        "before": before_count,
        "after": after_count,
        "groups": len(alias_groups),
        "merged": merged_count,
        "elapsed": elapsed,
    }
