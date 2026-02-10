"""Pass 5: Meeting briefing — synthesize upcoming meeting context from the knowledge graph.

Architecture:
  1. Query upcoming calendar events from graph.db (next N days)
  2. Classify events (professional / personal / holiday)
  3. Deduplicate similar events (e.g. two Valentine's Day entries)
  4. For each actionable meeting, walk the graph via high-weight edges:
     - calendar → attendee → contact → sent_to/cc_to → emails/messages
     - calendar → same_event/same_thread → related nodes
     - calendar → same_entity (high weight) → related context
  5. Search derived.db for open commitments involving attendees
  6. Optionally web-search for external context
  7. Synthesize via LLM using PDB/BLUF-inspired prompt
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rich.console import Console

from alteris_listener.graph.derived_store import DerivedStore
from alteris_listener.graph.store import GraphStore
from alteris_listener.llm.client import LLMClient

logger = logging.getLogger(__name__)
console = Console()


# ══════════════════════════════════════════════════════════════════
# System prompts (static — no format strings)
# ══════════════════════════════════════════════════════════════════

BRIEFING_SYSTEM_PROMPT = """\
You are an AI chief of staff. Your job is to make the user fully prepared for every event \
on their calendar.

CORE PRINCIPLE: For each event, put yourself in the user's shoes. You are about to walk \
into this meeting / event / dinner. Ask yourself:
- What do I need to know to succeed here?
- What should I bring?
- What could go wrong, and how do I prevent it?
- What coordination do I need to have done beforehand?
- What's the one thing that would make this go better than expected?

"Success" is contextual. For a recruiting call, success = knowing who they are, what they \
want, and having your own ask ready. For a kid's birthday party, success = arriving on time \
with a good gift, parking sorted, and childcare handled. For a team sync, success = knowing \
what happened this week and what's blocking people. For a client dinner, success = knowing \
the account status, allergies, and having the restaurant confirmed.

You will receive structured context: meeting metadata, related emails/messages/meetings from \
the user's knowledge graph, open commitments, web search results, and a user profile with \
ambient context (home location, family, preferences, local knowledge).

FORMAT — for each event:

## [Event Title]
**When:** [date, time, duration]  |  **Where:** [location or link]
**With:** [attendee names and roles/affiliations if known]

**BLUF:** [One sentence. What this is, why it matters, and the single most important thing \
to know or do. If you read only this line, you're 80% prepared.]

Then include ONLY the sections that are relevant. Not every event needs every section. \
Pick from:

**Their agenda:** What the other party wants from you. Cite evidence (emails, messages).
**Your agenda:** What you should want from this. Infer from commitments and context.
**What changed:** Developments since last interaction. Omit if nothing changed.
**Open items:** Active commitments with status/deadlines. Only if they exist.
**People:** Who's in the room, their org, your history. Only for external or unfamiliar contacts.
**Gift ideas:** For events you're invited to (birthdays, parties). Suggest 2-3 specific items \
with prices and where to buy locally. Use the event theme, honoree's age, venue as signals. \
Tier recommendations by budget (affordable → mid → premium).
**Logistics:** Travel time, parking, weather, what to bring, reservation numbers. \
If the user has dependents whose schedules are affected, flag coordination needs. \
If the user's profile mentions travel preferences, apply them.
**Prep:** 1-3 concrete actions to take before the event. Specific, not vague.

---

For HOLIDAYS:
- If the holiday has NO impact based on the context: single line.
  📌 **[Holiday Name]** — [date].
- If the context contains signals that the holiday affects the user (closure emails from \
schools/daycares/offices, travel bookings, pet boarding, schedule changes, etc.), expand:
  📌 **[Holiday Name]** — [date].
  **Logistics:** [What's affected, what needs attention, what actions to take.]
  If dependents' care is affected by closures and no arrangements appear in context, \
  flag it and suggest contacting care providers. Draft a ready-to-send message if you \
  know the provider's name from the user profile.
  Think broadly: holidays affect schools, daycares, banks, offices, medical appointments, \
  travel patterns, and local businesses. What matters depends on who this user is.

CROSS-EVENT INTELLIGENCE:
After all briefings, add if applicable:

## ⚡ Cross-Event Notes
Connections, conflicts, or compound logistics across events. Examples:
- "Back-to-back calls — end the first by 11:00 to prep for the second"
- "Both meetings touch the same hiring pipeline — coordinate messaging"
- "Party ends at 3:30, dinner at 4:00 — 15 min drive, tight but doable"

RULES:
1. BLUF first. Always. Reader may stop there.
2. Never fabricate. If context is thin: "Limited prior context."
3. Cite evidence. "Per [Name]'s [date] email" not "They previously reached out."
4. Recency wins. Last 2 weeks > older context.
5. STALENESS AWARENESS: If the triage flagged a strategic shift (pivot, reorg, new \
   direction), the briefing MUST lead with the current state, not the old plan. \
   Pre-shift context should be mentioned only to show what changed: \
   "Previously focused on X (per Jan 15 thread), but pivoted to Y after [event]." \
   Don't present stale priorities as current action items.
6. Flag urgency with ⚠️ for overdue, blocking, or time-sensitive items.
7. Be terse. No filler. No "here's your briefing" preamble.
8. Think about failure modes. What could go wrong? Traffic, weather, missing info, \
   forgotten commitments, double-bookings, no gift, wrong location.
9. One page per event max. Prioritize ruthlessly.
10. Collapse duplicates across calendars.
11. Use the User Profile to personalize. If it mentions travel preferences, dependents, \
    local knowledge, or care providers, weave them into logistics naturally."""


WEB_SEARCH_SYSTEM_PROMPT = """\
You are a research assistant that generates web search queries to help prepare for meetings and events.

Given a meeting title, attendees, classification, and a brief context summary, \
generate 1-3 focused search queries. Focus on:
- Attendee backgrounds (LinkedIn, company roles) if they're external contacts
- Company news or updates relevant to the meeting topic
- Technical topics being discussed that may have recent developments
- Logistics: local events that could affect traffic, venue parking, weather alerts

Return [] (empty array) for:
- Internal team meetings (professional_internal) unless they mention external topics
- Holidays and observances
- Plain personal events (own dinners, reservations)
- Meetings where the existing context is already sufficient

Respond with ONLY a valid JSON array of strings. No markdown, no explanation."""


TRIAGE_SYSTEM_PROMPT = """\
You are an intelligence analyst reviewing raw context gathered for an upcoming meeting briefing. \
Your job is to assess what we have, request what we're missing, and detect if the landscape has shifted.

You will receive:
- Meeting metadata (title, time, attendees, classification)
- Raw context pulled from the user's knowledge graph (emails, messages, past meetings, commitments)

CRITICAL — DETECT STRATEGIC SHIFTS:
Before triaging individual items, scan for evidence that priorities, direction, or strategy \
changed recently. Signals include:
- "We decided to...", "new direction", "pivot", "let's focus on X instead"
- A brainstorm or offsite that produced new priorities
- A funding decision (pass, term sheet, pivot to revenue) that changes everything
- A reorg, departure, or new hire that reshuffles responsibilities
- A product decision that makes previous work irrelevant

SCOPE RULE: A shift must be directly relevant to THIS SPECIFIC MEETING to be flagged. \
People have multiple independent threads in their lives — a product architecture change \
at the user's company is separate from their personal career exploration. A shift in one \
domain does not make context in another domain stale. Ask yourself: "Does this shift change \
what the user should do or say IN THIS MEETING?" If not, don't flag it.

Examples:
- Team call context shows old product roadmap + evidence of a product pivot → flag it. \
  The old roadmap items are stale for this meeting.
- Team call context shows the user is also job searching → do NOT flag as a shift for the \
  team call. The job search is a separate thread that doesn't change team priorities.
- 1:1 with a recruiter + evidence the user changed target roles → flag it. Old role prep \
  is stale for this meeting.

If you detect a shift RELEVANT TO THIS MEETING:
- Note it in the assessment with approximate date
- Mark pre-shift context as potentially stale (still include it, but flag it)
- Request the specific thread/meeting where the decision happened if it's not in context
- The briefing MUST reflect the current state, not the old plan

Respond with a JSON object containing exactly these fields:

{
  "relevant_ids": ["id1", "id2", ...],
  "irrelevant_ids": ["id3", "id4", ...],
  "stale_ids": ["id5", "id6", ...],
  "detected_shift": {
    "found": true,
    "description": "Team pivoted from enterprise sales to PLG model after Khosla pass (approx Jan 24)",
    "pivot_date_approx": "2025-01-24"
  },
  "missing": [
    {
      "query_type": "person_emails",
      "params": {"name": "Jane Rivera", "email": "jane@acmecorp.com"},
      "reason": "Need the original outreach email to understand what partnership she's proposing"
    },
    {
      "query_type": "topic_search",
      "params": {"keywords": ["Q4 roadmap", "product launch"]},
      "reason": "Meeting is about the product timeline — need context on recent discussions"
    },
    {
      "query_type": "recent_from_person",
      "params": {"name": "David Chen", "days": 14},
      "reason": "David was mentioned in the last meeting — need to see if his follow-up arrived"
    },
    {
      "query_type": "commitments_search",
      "params": {"keywords": ["proposal", "contract", "deadline"]},
      "reason": "Check if there are open commitments related to this vendor relationship"
    }
  ],
  "assessment": "We have the vendor's initial pitch emails but lack context on the internal \
decision. Need David's follow-up thread to understand where the evaluation stands."
}

NOTES:
- "stale_ids": items that are pre-shift and may no longer reflect current priorities. \
  These will still be included in context but the synthesis LLM will know to treat them \
  with caution. If no shift detected, return [].
- "detected_shift": set "found" to false if no shift detected. This field helps the \
  synthesis LLM understand the strategic landscape.

QUERY TYPES available:
- person_emails: Find emails sent to/from a specific person (by name or email)
- person_messages: Find iMessages/Slack messages involving a person
- topic_search: Search nodes by keyword in subject/body_preview
- recent_from_person: Get the N most recent communications with a person
- commitments_search: Search open commitments by keyword
- thread_lookup: Get full thread by thread_id
- meeting_lookup: Get full meeting notes by node ID (use for items in the "Additional Past Meetings" manifest)

RULES:
1. Be surgical. Only request what would materially improve the briefing.
2. Max 5 missing requests per meeting. Prioritize ruthlessly.
3. Mark IDs as irrelevant if they clearly don't relate to this meeting.
4. If context is already sufficient, return empty missing array.
5. The assessment should be 1-2 sentences: what we know, what gap matters most.
6. If you detect a strategic shift, your FIRST missing request should be for the thread \
   or meeting where the decision was made — if it's not already in context."""


QUESTIONS_SYSTEM_PROMPT = """\
You are an AI chief of staff reviewing the prepared context for a set of upcoming events. \
Your job is to identify what YOU CANNOT KNOW from the knowledge graph alone — things only \
the user can tell you — that would materially improve the briefing quality.

Think like a great executive assistant: before your boss walks into a meeting, you'd ask \
"Do you want me to push back on the deadline?" or "Should I book a car for after?" \
You wouldn't ask about things you can already see in the data.

For each event, generate 0-3 questions. Each question should:
1. Be specific to the event (not generic)
2. Address something the graph CANNOT answer (user preferences, decisions, undocumented logistics)
3. Have a clear impact on the briefing if answered

Categories of useful questions:
- LOGISTICS: "Do you have childcare/pet care arranged?" / "Is your partner joining?"
- DECISIONS: "Are you interested in this role or just exploring?" / "Do you want to push back on the deadline or accept it?"
- MISSING CONTEXT: "Were the Q3 metrics fixed? I see your manager flagged them but no follow-up."
- MATERIALS: "Do you have an updated CV/deck/proposal? I can tailor talking points."
- COORDINATION: "Should I flag the scheduling conflict to anyone?" / "Does your partner know about the tight transition between events?"

Respond with a JSON object:
{
  "questions": [
    {
      "event_subject": "Screening Call with Recruiter",
      "question": "Are you genuinely interested in this role, or exploring to keep options open? This changes whether I'd prep probing questions vs. polite information-gathering.",
      "category": "DECISIONS",
      "context_if_answered": "Will tailor the briefing's 'Your agenda' section and talking points",
      "accepts_file": false
    },
    {
      "event_subject": "Screening Call with Recruiter",
      "question": "Do you have a current CV or resume for this application? I can analyze it against the role description and identify gaps or highlights.",
      "category": "MATERIALS",
      "context_if_answered": "Will analyze the document and generate role-specific prep",
      "accepts_file": true
    },
    {
      "event_subject": "Birthday party at venue",
      "question": "Are your kids coming to the party or do they need separate care? This affects logistics and timing.",
      "category": "LOGISTICS",
      "context_if_answered": "Will add care coordination and adjust travel calculations"
    },
    {
      "event_subject": "Dinner reservation",
      "question": "Any gift or flowers planned for your partner? I can suggest pickup stops along your route.",
      "category": "LOGISTICS",
      "context_if_answered": "Will add gift pickup stop to the logistics timeline"
    }
  ]
}

QUESTION SELECTION — use the maximum information gain principle:
A good question is one where the answer splits your uncertainty roughly in half. \
If you can already predict the answer with >80% confidence, don't ask — just go with \
your best guess. If the answer wouldn't change what you write, don't ask.

For each candidate question, apply this test:
  - Imagine answer A (e.g. "yes, interested in the role")
  - Imagine answer B (e.g. "no, just networking")
  - Would the briefing differ meaningfully between A and B?
  - Is your prior roughly 50/50 between A and B?
If BOTH conditions hold, ask. If you're already 80% sure of the answer, or if both \
answers lead to the same briefing, skip it.

Examples of HIGH information gain:
  - "Are you interested in this role or just keeping options open?" → completely \
changes whether you prep probing questions vs. polite deflection. Prior is ~50/50.
  - "Are your kids coming to the event or do they need separate care?" → changes \
logistics, what to bring, travel time. Can't predict from context.

Examples of LOW information gain (DON'T ASK):
  - "Do you want me to include weather info?" → yes is obvious, just include it.
  - "Are you planning to attend this meeting?" → it's on the calendar, prior >95% yes.
  - "Do you want talking points?" → always yes for a screening call.

RULES:
1. Apply the information gain test above. Typically yields 0-3 questions. Zero is common.
2. Don't ask about things visible in the context (attendees, times, locations).
3. Don't ask when you can make a safe default assumption.
4. Group related questions — don't ask 3 questions about the same logistics issue.
5. For materials questions (CV, documents, reports), set accepts_file: true.
6. If all events have sufficient context, return {"questions": []}."""


# ══════════════════════════════════════════════════════════════════
# User identity — derived from graph or profile
# ══════════════════════════════════════════════════════════════════

def _get_user_profile_path() -> Path:
    """Return path to user profile config."""
    return Path.home() / ".alteris" / "profile.yaml"


def _load_user_profile() -> dict:
    """Load user profile from ~/.alteris/profile.yaml if it exists.

    Expected format:
        name: "Jane Smith"
        timezone: "America/New_York"
        emails:
          - jane@company.com
          - janesmith@gmail.com
        home:
          neighborhood: "Capitol Hill"
          city: "Seattle"
          state: "WA"
        family:
          spouse: "Alex Smith"
          children:
            - name: "Sam"
              school: "Stevens Elementary"
            - name: "Maya"
              daycare: "KinderCare"
          care_providers:
            - name: "Maria"
              role: "Primary babysitter"
            - name: "Grandma Sue"
              role: "Backup, weekdays"
          pets:
            - name: "Max"
              type: "dog"
              daycare: "Camp Bow Wow"
              walker: "Rover - Sarah K."
        travel:
          airline_loyalty: "Alaska Airlines"
          seat_preference: "aisle"
          class_preference: "economy plus"
          hotel_loyalty: "Marriott Bonvoy"
          rental_car: "prefers not to rent"
        local_knowledge:
          - "Trader Joe's has affordable flowers ($5-10)"
          - "Parking on Broadway is terrible on weekends"

    TODO — graph-inferred profile generation:
    The entire profile.yaml should eventually be auto-generated from graph data,
    not manually authored. The graph contains enough signal to infer every field:

    Identity & basics:
      - name: most frequent sender name in outbound emails
      - emails: already inferred from sender frequency (see _get_user_emails)
      - timezone: infer from calendar event timestamps or email send-time patterns
      - home: infer from most frequent evening/weekend locations in calendar events,
        or from delivery addresses in e-commerce confirmation emails
      - role: extract from email signature blocks or LinkedIn profile if in graph

    Family & dependents:
      - spouse: most frequent co-attendee on personal calendar events; frequent
        evening/weekend message contact; shared last name on school correspondence
      - children: detect from school email senders (e.g. "noreply@seattleschools.org"),
        daycare correspondence (e.g. "brightside@brighthorizons.com"), pediatrician
        appointment confirmations. Names from correspondence subjects/bodies.
      - care_providers: detect from frequent evening/weekend message contacts who
        aren't family; babysitter booking confirmations; Venmo/payment patterns
      - pets: detect from vet appointment emails, pet service senders (Rover, Wag),
        pet food subscription emails, boarding confirmations

    Travel preferences:
      - airline_loyalty: most frequent airline in booking confirmation emails
      - seat_preference: extract from confirmation emails ("Seat 14C - aisle")
      - class_preference: fare class from booking confirmations ("Main Cabin", "Economy Plus")
      - hotel_loyalty: most frequent hotel chain in booking confirmations
      - rental_car: presence/absence of rental car booking patterns

    Local knowledge:
      - Frequent restaurant reservations (OpenTable, Resy confirmations)
      - Grocery store patterns (receipt emails from specific stores)
      - Parking complaints or tips in messages
      - Commute patterns from calendar event locations vs home

    Implementation approach: a dedicated `graph profile-infer` CLI command that
    scans the graph, runs heuristics + an LLM synthesis pass, and writes/updates
    ~/.alteris/profile.yaml. User reviews and corrects. Re-run periodically to
    pick up changes (new school, moved house, new babysitter).
    """
    profile_path = _get_user_profile_path()
    if not profile_path.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(profile_path.read_text()) or {}
    except ImportError:
        # Fallback: try json
        try:
            return json.loads(profile_path.read_text())
        except Exception:
            return {}
    except Exception:
        return {}


# Cache for user identity lookups
_user_identity_cache: dict = {}


def _get_user_emails(store: GraphStore | None = None) -> set[str]:
    """Get the user's email addresses from profile or graph heuristics."""
    if "emails" in _user_identity_cache:
        return _user_identity_cache["emails"]

    emails = set()

    # Source 1: profile.yaml
    profile = _load_user_profile()
    if profile.get("emails"):
        emails.update(e.lower() for e in profile["emails"])

    # Source 2: graph heuristic — emails that appear most frequently as sender
    if store and not emails:
        try:
            rows = store.conn.execute(
                """SELECT sender, COUNT(*) as cnt FROM nodes
                   WHERE node_type = 'email' AND sender IS NOT NULL
                   GROUP BY sender ORDER BY cnt DESC LIMIT 5"""
            ).fetchall()
            # The user's own email is typically the most frequent sender
            for r in rows:
                sender = (r[0] or "").lower()
                if "@" in sender and r[1] > 20:  # at least 20 sent emails
                    emails.add(sender)
        except Exception:
            pass

    _user_identity_cache["emails"] = emails
    return emails


def _get_user_domains(store: GraphStore | None = None) -> set[str]:
    """Get the user's email domains (for internal/external classification)."""
    if "domains" in _user_identity_cache:
        return _user_identity_cache["domains"]

    emails = _get_user_emails(store)
    domains = set()
    for e in emails:
        if "@" in e:
            domain = e.split("@")[-1]
            domains.add(domain)

    # Common personal email domains are always "internal" (not external contacts)
    domains.update({"gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
                    "icloud.com", "me.com", "protonmail.com"})

    _user_identity_cache["domains"] = domains
    return domains


# ══════════════════════════════════════════════════════════════════
# Event classification and dedup
# ══════════════════════════════════════════════════════════════════

# Keywords that indicate holidays/observances (case-insensitive)
_HOLIDAY_KEYWORDS = {
    "valentine's day", "presidents' day", "president's day", "presidents day",
    "mlk day", "martin luther king", "memorial day", "independence day",
    "labor day", "thanksgiving", "christmas", "new year", "easter",
    "juneteenth", "veterans day", "columbus day", "indigenous peoples",
    "halloween", "good friday", "observance",
}

_PERSONAL_KEYWORDS = {
    "birthday", "reservation:", "dinner", "brunch", "lunch with",
    "date night", "anniversary", "party",
}


def _classify_event(event: dict, store: GraphStore | None = None) -> str:
    """Classify a calendar event into a tier.

    Returns one of:
      - 'holiday': all-day holidays/observances -> collapsed to one line
      - 'personal': birthdays, reservations, social -> short format
      - 'personal_external': events from external invitations (evites, party invites)
      - 'professional_internal': team meetings, 1:1s with coworkers
      - 'professional_external': recruiting calls, investor meetings, etc.
    """
    subject = (event.get("subject") or "").lower()
    data = event.get("_data", {})
    attendees = event.get("_attendees", [])

    # All-day events with no attendees and holiday keywords -> holiday
    is_all_day = data.get("all_day", False)
    if not is_all_day and event.get("_local_dt"):
        dt = event["_local_dt"]
        if dt.hour == 0 and dt.minute == 0 and len(attendees) == 0:
            is_all_day = True

    for kw in _HOLIDAY_KEYWORDS:
        if kw in subject:
            return "holiday"

    # ── Detect external invitations (evites, party platforms, etc.) ──
    # Check if the event was created by an external organizer
    organizer_email = (data.get("organizer_email") or data.get("organizer", {}).get("email", "")
                       if isinstance(data.get("organizer"), dict) else data.get("organizer_email", ""))
    organizer_email = (organizer_email or "").lower()
    # TODO: Load user_emails from ~/.alteris/profile.yaml
    user_emails = _get_user_emails(store)

    is_external_invite = False
    if organizer_email and organizer_email not in user_emails:
        is_external_invite = True

    # Evite signals: event body contains evite/partiful/paperless post URLs
    body = (data.get("body") or event.get("body_preview") or "").lower()
    evite_domains = ["evite.com", "partiful.com", "paperlesspost.com", "punchbowl.com",
                     "greenvelope.com", "signup.com"]
    has_evite = any(domain in body for domain in evite_domains)
    if has_evite:
        is_external_invite = True

    # ── Personal keyword detection ──
    is_personal = False
    for kw in _PERSONAL_KEYWORDS:
        if kw in subject:
            is_personal = True
            break

    # If personal AND external invite → personal_external (gift-worthy)
    if is_personal and is_external_invite:
        return "personal_external"

    # Heuristic: birthday party with a name + venue + age → likely external invite
    # (if it's your own child, you'd typically not put the age in the subject)
    if is_personal and "birthday" in subject:
        import re
        # Check for "[Name]'s [Nth] birthday" pattern
        name_match = re.search(r"^(.+?)(?:'s|'s)\s", subject)
        if name_match:
            party_name = name_match.group(1).strip().lower()
            # If the event has a location (venue) → it's a party, not just a reminder
            has_venue = bool(data.get("location"))
            # If it mentions an age ("6th birthday") → party invitation
            has_age = bool(re.search(r"\d+(?:st|nd|rd|th)\s*birthday", subject, re.IGNORECASE))
            if has_venue and has_age:
                return "personal_external"

    if is_personal:
        return "personal"

    # ── Professional classification via email domains ──
    user_domains = _get_user_domains(store)
    has_external = False
    for a in attendees:
        email = (a.get("email") or "").lower()
        if email:
            domain = email.split("@")[-1] if "@" in email else ""
            if domain and domain not in user_domains:
                has_external = True
                break

    if has_external:
        return "professional_external"

    # ── Fallback: check graph for professional signals ──
    # If attendees have names but no emails, look up their contact nodes
    if store and attendees:
        for a in attendees:
            name = (a.get("name") or "").lower()
            if not name or len(name) < 3:
                continue
            # Search contact_stats for this person
            rows = store.conn.execute(
                """SELECT contact_id, importance_tier, total_messages
                   FROM contact_stats
                   WHERE LOWER(display_name) LIKE ?
                   LIMIT 3""",
                (f"%{name}%",),
            ).fetchall()
            for r in rows:
                contact_id = r[0]
                # If their contact ID has an external domain, it's professional
                if "@" in contact_id:
                    domain = contact_id.split("@")[-1]
                    if domain not in user_domains:
                        return "professional_external"

    # ── Subject-based professional signals ──
    professional_keywords = {
        "call with", "intro call", "screening", "interview", "1:1",
        "sync", "standup", "sprint", "retro", "planning",
        "investor", "fundrais", "pitch", "demo",
    }
    for kw in professional_keywords:
        if kw in subject:
            if attendees:
                return "professional_external"
            # No attendees but professional keyword → still likely professional
            return "professional_external"

    if attendees:
        return "professional_internal"

    if is_all_day:
        return "holiday"

    if is_external_invite:
        return "personal_external"

    return "personal"


def _dedup_events(events: list[dict]) -> list[dict]:
    """Collapse duplicate events (e.g. two Valentine's Day entries from different calendars)."""
    import re

    def _normalize_subject(s: str) -> str:
        """Aggressively normalize for dedup matching."""
        s = s.strip().lower()
        # Strip unicode variants (curly quotes, etc)
        s = s.replace("\u2019", "'").replace("\u2018", "'")
        # Remove trailing periods, extra whitespace
        s = re.sub(r"[.\s]+$", "", s)
        s = re.sub(r"\s+", " ", s)
        return s

    seen = {}
    result = []
    for e in events:
        subj = _normalize_subject(e.get("subject") or "")
        date_key = ""
        if e.get("_local_dt"):
            date_key = e["_local_dt"].strftime("%Y-%m-%d")
        key = (subj, date_key)

        if key in seen:
            existing = seen[key]
            if len(e.get("_attendees", [])) > len(existing.get("_attendees", [])):
                idx = result.index(existing)
                result[idx] = e
                seen[key] = e
        else:
            # Also check for near-duplicates: same date + both classified as holiday
            dup_found = False
            if e.get("_classification") == "holiday":
                for existing_key, existing_event in seen.items():
                    if existing_key[1] == date_key and existing_event.get("_classification") == "holiday":
                        # Two holidays on the same day → likely duplicates
                        dup_found = True
                        break
            if not dup_found:
                seen[key] = e
                result.append(e)
    return result


# ══════════════════════════════════════════════════════════════════
# Calendar event discovery
# ══════════════════════════════════════════════════════════════════

def _get_upcoming_events(
    store: GraphStore,
    days_ahead: int = 7,
    user_tz: str = "America/Los_Angeles",
) -> list[dict]:
    """Query calendar_event nodes within the next N days."""
    import msgpack
    import zoneinfo

    local_tz = zoneinfo.ZoneInfo(user_tz)
    now_local = datetime.now(local_tz)
    start_ts = int(now_local.timestamp())
    end_ts = int((now_local + timedelta(days=days_ahead)).timestamp())

    rows = store.conn.execute(
        """SELECT id, node_type, source, timestamp, subject, sender,
                  recipients, body_preview, data, thread_id,
                  triage_relevant, triage_adjusted
           FROM nodes
           WHERE source = 'calendar'
             AND timestamp >= ?
             AND timestamp <= ?
           ORDER BY timestamp ASC""",
        (start_ts, end_ts),
    ).fetchall()

    events = []
    for row in rows:
        event = dict(row)
        if event.get("data"):
            try:
                event["_data"] = msgpack.unpackb(event["data"], raw=False)
            except Exception:
                event["_data"] = {}
        else:
            event["_data"] = {}

        # Parse attendees
        data = event["_data"]
        attendees = []
        if data.get("attendees"):
            raw_att = data["attendees"]
            if isinstance(raw_att, list):
                for a in raw_att:
                    if isinstance(a, dict):
                        attendees.append({
                            "name": a.get("name", ""),
                            "email": a.get("email", ""),
                        })
                    elif isinstance(a, str):
                        attendees.append({"name": a, "email": ""})
        event["_attendees"] = attendees

        # Local time
        if event["timestamp"]:
            dt = datetime.fromtimestamp(event["timestamp"], tz=local_tz)
            event["_local_time"] = dt.strftime("%a %b %d, %Y %I:%M %p %Z")
            event["_local_dt"] = dt
        else:
            event["_local_time"] = "(unknown time)"
            event["_local_dt"] = None

        # Classify (pass store for graph-based contact lookup)
        event["_classification"] = _classify_event(event, store=store)
        events.append(event)

    # Dedup
    events = _dedup_events(events)

    return events


# ══════════════════════════════════════════════════════════════════
# Graph context gathering (per meeting) — multi-hop via attendee edges
# ══════════════════════════════════════════════════════════════════

def _gather_meeting_context(
    event: dict,
    store: GraphStore,
    derived: DerivedStore,
    max_emails: int = 10,
    max_meetings: int = 5,
    max_messages: int = 6,
    user_tz: str = "America/Los_Angeles",
) -> dict:
    """Walk the graph from a calendar event to gather related context.

    Strategy (multi-hop with weight gates):
      Hop 0: The calendar event node itself
      Hop 1 (structural, any weight):
        - same_event -> linked meeting notes, email invites
        - same_thread -> conversation thread siblings
        - attendee -> contact nodes for each attendee
      Hop 2a (from contacts, any weight):
        - sent_to / cc_to / reply_to -> emails involving the contact
        - attendee -> other calendar events with this person
      Hop 2b (from non-contact hop-1 nodes, weight-gated):
        - same_entity (weight >= 0.30) -> only strong entity matches
        - same_topic (weight >= 0.15) -> only strong topic matches

    Returns a dict with:
      - related_emails: list of dicts (subject, sender, date, body_preview)
      - related_meetings: list of dicts (past meetings with same people)
      - related_messages: list of dicts (iMessages with attendees)
      - commitments: list of dicts (open commitments involving attendees)
      - context_summary: short text summary for logging
    """
    import msgpack
    import zoneinfo

    local_tz = zoneinfo.ZoneInfo(user_tz)
    event_id = event["id"]

    def _load_node(node_id):
        """Load a node and unpack its data blob."""
        row = store.conn.execute(
            """SELECT id, node_type, source, timestamp, subject, sender,
                      recipients, body_preview, data
               FROM nodes WHERE id = ?""",
            (node_id,),
        ).fetchone()
        if not row:
            return None
        node = dict(row)
        if node.get("data"):
            try:
                node["_data"] = msgpack.unpackb(node["data"], raw=False)
            except Exception:
                node["_data"] = {}
        else:
            node["_data"] = {}
        return node

    def _get_neighbors(node_id, edge_type, min_weight=0.0):
        """Get neighbor IDs and weights for a given edge type."""
        return store.conn.execute(
            """SELECT dst, weight FROM edges
               WHERE src = ? AND edge_type = ? AND weight >= ?
               UNION
               SELECT src, weight FROM edges
               WHERE dst = ? AND edge_type = ? AND weight >= ?""",
            (node_id, edge_type, min_weight, node_id, edge_type, min_weight),
        ).fetchall()

    # ── Relevance-scored BFS: walk graph, score by edge_weight / hop ──
    import time as _time

    now = _time.time()
    visited = {event_id}
    hop1_contacts = []
    node_scores: dict[str, float] = {}
    node_meta: dict[str, dict] = {}

    def _recency_boost(node):
        """Boost score by time bucket: this week > last week > ... > older."""
        ts = node.get("timestamp", 0) if node else 0
        if not ts:
            return 1.0
        age_days = (now - ts) / 86400
        if age_days <= 7:       # this week
            return 3.0
        elif age_days <= 14:    # last week
            return 2.0
        elif age_days <= 30:    # this month
            return 1.5
        elif age_days <= 90:    # last 3 months
            return 1.2
        elif age_days <= 365:   # last year
            return 1.0
        return 0.7              # older than a year

    def _record(nid, node, hop, weight, via_edge, via_contact=None):
        score = (weight / hop) * _recency_boost(node)
        if nid in node_scores:
            if score > node_scores[nid]:
                node_scores[nid] = score
                node_meta[nid] = {"hop": hop, "via_edge": via_edge,
                                  "via_contact": via_contact, "node": node}
        else:
            node_scores[nid] = score
            node_meta[nid] = {"hop": hop, "via_edge": via_edge,
                              "via_contact": via_contact, "node": node}

    # Hop 1: structural edges from calendar event
    HOP1_EDGE_TYPES = ["same_event", "same_thread", "attendee"]
    for etype in HOP1_EDGE_TYPES:
        for nbr_id, weight in _get_neighbors(event_id, etype):
            if nbr_id in visited:
                continue
            visited.add(nbr_id)
            node = _load_node(nbr_id)
            if not node:
                continue
            if node["node_type"] == "contact":
                hop1_contacts.append(node)
            else:
                _record(nbr_id, node, 1, weight, etype)

    # Hop 2a: from contacts, follow structural edges
    CONTACT_EDGE_TYPES = ["sent_to", "cc_to", "reply_to", "attendee"]
    for contact in hop1_contacts:
        cid = contact["id"]
        contact_name = contact.get("_data", {}).get("display_name", cid)
        for etype in CONTACT_EDGE_TYPES:
            for nbr_id, weight in _get_neighbors(cid, etype):
                already_seen = nbr_id in visited
                if not already_seen:
                    visited.add(nbr_id)
                    node = _load_node(nbr_id)
                    if not node or node["node_type"] == "contact":
                        continue
                    _record(nbr_id, node, 2, weight, f"contact:{etype}", contact_name)
                elif nbr_id in node_meta:
                    _record(nbr_id, node_meta[nbr_id]["node"], 2,
                            weight, f"contact:{etype}", contact_name)

    # Hop 2b + Hop 3: semantic edges from content nodes
    SEMANTIC_EDGES = {"same_entity": 0.05, "same_topic": 0.05}
    for hop_level in (1, 2):
        source_ids = [nid for nid, m in node_meta.items() if m["hop"] == hop_level]
        target_hop = hop_level + 1
        for src_id in source_ids:
            for etype, min_w in SEMANTIC_EDGES.items():
                for nbr_id, weight in _get_neighbors(src_id, etype, min_w):
                    already_seen = nbr_id in visited
                    if not already_seen:
                        visited.add(nbr_id)
                        node = _load_node(nbr_id)
                        if not node or node["node_type"] == "contact":
                            continue
                        _record(nbr_id, node, target_hop, weight, etype)
                    elif nbr_id in node_meta:
                        _record(nbr_id, node_meta[nbr_id]["node"], target_hop,
                                weight, etype)

    # ── Rank and split into full-content vs manifest ──
    ranked = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
    seven_days_ago = _time.time() - 7 * 86400

    related_emails = []
    related_meetings = []
    related_messages = []
    meeting_manifest = []
    email_count = meeting_count = message_count = 0

    for nid, score in ranked:
        meta = node_meta[nid]
        node = meta["node"]
        if not node:
            continue
        node_type = node["node_type"]

        ts_str = ""
        if node.get("timestamp"):
            try:
                dt = datetime.fromtimestamp(node["timestamp"], tz=local_tz)
                ts_str = dt.strftime("%Y-%m-%d %H:%M")
            except (OSError, ValueError):
                pass

        is_recent = node.get("timestamp", 0) >= seven_days_ago

        base = {
            "id": nid,
            "subject": node.get("subject", "(no subject)"),
            "sender": node.get("sender", ""),
            "date": ts_str,
            "hop": meta["hop"],
            "via": meta["via_edge"],
            "relevance": round(score, 3),
        }

        if node_type == "email":
            if email_count < max_emails:
                base["body_preview"] = (node.get("body_preview") or "")[:400]
                related_emails.append(base)
                email_count += 1

        elif node_type == "meeting":
            meeting_body = node.get("_data", {}).get("body", "") or node.get("body_preview", "")
            if is_recent or meeting_count < max_meetings:
                base["body_preview"] = (node.get("body_preview") or "")[:400]
                base["notes_preview"] = (meeting_body or "")[:1500]
                related_meetings.append(base)
                meeting_count += 1
            else:
                meeting_manifest.append(base)

        elif node_type == "message":
            if message_count < max_messages:
                base["body_preview"] = (node.get("body_preview") or "")[:400]
                related_messages.append(base)
                message_count += 1

        elif node_type == "calendar" and nid != event_id:
            if is_recent or meeting_count < max_meetings:
                related_meetings.append(base)
                meeting_count += 1

    # Sort each list by date descending
    for lst in (related_emails, related_meetings, related_messages):
        lst.sort(key=lambda x: x.get("date", ""), reverse=True)

    # ── Find open commitments involving attendees ──
    attendee_names_full = [a["name"].lower() for a in event["_attendees"] if a.get("name")]
    attendee_names_first = [a["name"].lower().split()[0] for a in event["_attendees"]
                            if a.get("name") and len(a["name"].split()[0]) > 2]
    attendee_emails = [a["email"].lower() for a in event["_attendees"] if a.get("email")]
    # Subject keywords for broader matching
    subject_words = set()
    for word in (event.get("subject") or "").lower().split():
        if len(word) > 3 and word not in {"with", "call", "meeting", "weekly", "team", "and"}:
            subject_words.add(word)

    commitments = []
    if attendee_names_first or attendee_emails or subject_words:
        all_open = derived.conn.execute(
            "SELECT * FROM commitments WHERE status = 'open'"
        ).fetchall()
        for c in all_open:
            c = dict(c)
            c_text = f"{c.get('who', '')} {c.get('to_whom', '')} {c.get('what', '')} {c.get('note', '')}".lower()
            match = False

            # Full name match (highest confidence)
            for name in attendee_names_full:
                if name and name in c_text:
                    match = True
                    break

            # First name match
            if not match:
                for name in attendee_names_first:
                    if name and name in c_text:
                        match = True
                        break

            # Email match
            if not match:
                for email in attendee_emails:
                    if email and email in c_text:
                        match = True
                        break

            # Subject keyword match (lower confidence)
            if not match:
                for word in subject_words:
                    if word in c_text:
                        match = True
                        break

            if match:
                commitments.append({
                    "type": c["commitment_type"],
                    "what": c["what"],
                    "who": c.get("who", ""),
                    "to_whom": c.get("to_whom", ""),
                    "status": c["status"],
                    "deadline": c.get("deadline"),
                    "priority": c["priority"],
                    "note": (c.get("note") or "")[:250],
                })

    # ── Build context summary for logging ──
    parts = []
    if related_emails:
        parts.append(f"{len(related_emails)} related emails")
    if related_meetings:
        parts.append(f"{len(related_meetings)} past meetings")
    if meeting_manifest:
        parts.append(f"{len(meeting_manifest)} more available on request")
    if related_messages:
        parts.append(f"{len(related_messages)} message threads")
    if commitments:
        parts.append(f"{len(commitments)} open commitments")
    context_summary = ", ".join(parts) if parts else "no related context found"

    return {
        "related_emails": related_emails,
        "related_meetings": related_meetings,
        "related_messages": related_messages,
        "meeting_manifest": meeting_manifest,
        "commitments": commitments,
        "contact_nodes": hop1_contacts,
        "context_summary": context_summary,
    }


# ══════════════════════════════════════════════════════════════════
# Web search agent
# ══════════════════════════════════════════════════════════════════

def _generate_search_queries(
    event: dict,
    context: dict,
    llm: LLMClient,
) -> list[str]:
    """Ask the LLM to suggest web search queries for this meeting."""
    attendee_str = ", ".join(
        a["name"] or a["email"] for a in event["_attendees"]
    ) or "(no attendees listed)"

    user_message = (
        f"Meeting: {event.get('subject', '(untitled)')}\n"
        f"Classification: {event.get('_classification', 'unknown')}\n"
        f"Attendees: {attendee_str}\n"
        f"Context: {context['context_summary']}"
    )

    result = llm.run_json(WEB_SEARCH_SYSTEM_PROMPT, user_message)

    if isinstance(result, list):
        return result[:3]
    if isinstance(result, dict) and "raw_text" in result:
        raw = result["raw_text"].strip()
        if raw.startswith("["):
            try:
                return json.loads(raw)[:3]
            except json.JSONDecodeError:
                pass
    return []


def _generate_gift_search_queries(event: dict) -> list[str]:
    """Generate gift-oriented web search queries based on event context.

    Uses signals from the event: child's name/age, venue, theme, event type.
    No LLM call needed — this is deterministic based on event metadata.
    """
    subject = event.get("subject", "")
    data = event.get("_data", {})
    body = data.get("body", "") or event.get("body_preview", "") or ""
    location = data.get("location", "")

    queries = []

    # Extract age if mentioned (e.g. "6th birthday", "turns 5")
    import re
    age_match = re.search(r"(\d+)(?:st|nd|rd|th)\s*birthday", subject, re.IGNORECASE)
    if not age_match:
        age_match = re.search(r"turns\s*(\d+)", subject + " " + body, re.IGNORECASE)
    age = age_match.group(1) if age_match else None

    # Extract name (first word of subject before "'s" or "birthday")
    name_match = re.search(r"^(.+?)(?:'s|'s)\s", subject)
    child_name = name_match.group(1).strip() if name_match else None

    # Venue-based gift signals
    venue_lower = location.lower() if location else ""
    subject_lower = subject.lower()
    combined = venue_lower + " " + subject_lower + " " + body.lower()
    theme = None
    if any(kw in combined for kw in ["cat", "kitten", "feline", "meow"]):
        theme = "cats"
    elif any(kw in combined for kw in ["dog", "puppy", "canine", "paw patrol"]):
        theme = "dogs"
    elif any(kw in combined for kw in ["trampoline", "bounce", "jump"]):
        theme = "trampoline park"
    elif any(kw in combined for kw in ["art studio", "paint", "pottery", "craft"]):
        theme = "arts and crafts"
    elif any(kw in combined for kw in ["swim", "pool", "water park", "splash"]):
        theme = "swimming"
    elif any(kw in combined for kw in ["gymnastics", "tumbl"]):
        theme = "gymnastics"
    elif any(kw in combined for kw in ["dinosaur", "dino", "museum"]):
        theme = "dinosaurs"
    elif any(kw in combined for kw in ["princess", "fairy", "unicorn"]):
        theme = "princess and unicorns"
    elif any(kw in combined for kw in ["space", "rocket", "astronaut", "planet"]):
        theme = "space"
    elif any(kw in combined for kw in ["lego", "building", "construction"]):
        theme = "LEGO and building"

    # Build queries
    if "birthday" in subject_lower:
        age_str = f"{age} year old" if age else "kids"
        if theme:
            queries.append(f"best {theme} themed birthday gifts for {age_str}")
            queries.append(f"popular {theme} toys for {age_str} 2025 2026")
        else:
            queries.append(f"best birthday gifts for {age_str}")

        # Local store query if we have a city from the location
        city = None
        if location:
            # Try to extract city from address
            city_match = re.search(r",\s*([A-Za-z\s]+),\s*[A-Z]{2}", location)
            if city_match:
                city = city_match.group(1).strip()
        if city and theme:
            queries.append(f"{theme} gifts toy store near {city}")
        elif city:
            queries.append(f"toy store near {city}")

    return queries[:3]


def _web_search(queries: list[str]) -> list[dict]:
    """Run web searches via SerpAPI.

    Returns list of {query, results: [{title, snippet, url}]}.
    Falls back gracefully if no search API is configured.
    """
    import os

    results = []

    api_key = os.environ.get("SERPAPI_KEY") or os.environ.get("GOOGLE_SEARCH_API_KEY")

    if not api_key:
        logger.debug("No search API key configured, skipping web search")
        return []

    try:
        import requests
    except ImportError:
        logger.debug("requests not installed, skipping web search")
        return []

    for query in queries:
        try:
            if os.environ.get("SERPAPI_KEY"):
                resp = requests.get(
                    "https://serpapi.com/search",
                    params={"q": query, "api_key": api_key, "num": 3},
                    timeout=10,
                )
                if resp.ok:
                    data = resp.json()
                    snippets = []
                    for r in data.get("organic_results", [])[:3]:
                        snippets.append({
                            "title": r.get("title", ""),
                            "snippet": r.get("snippet", ""),
                            "url": r.get("link", ""),
                        })
                    results.append({"query": query, "results": snippets})
            else:
                logger.debug("Skipping search for: %s (no supported API)", query)
        except Exception as e:
            logger.warning("Web search failed for '%s': %s", query, e)

    return results


# ══════════════════════════════════════════════════════════════════
# Agentic triage — LLM reviews context and requests refinements
# ══════════════════════════════════════════════════════════════════

def _execute_graph_query(
    query: dict,
    store: GraphStore,
    derived: DerivedStore,
    user_tz: str = "America/Los_Angeles",
) -> list[dict]:
    """Execute a single graph query requested by the triage LLM.

    Supports: person_emails, person_messages, topic_search,
              recent_from_person, commitments_search, thread_lookup
    """
    import msgpack
    import zoneinfo

    local_tz = zoneinfo.ZoneInfo(user_tz)
    qtype = query.get("query_type", "")
    params = query.get("params", {})
    results = []

    try:
        if qtype == "person_emails":
            # Find emails involving a person by name or email
            name = (params.get("name") or "").lower()
            email = (params.get("email") or "").lower()
            search_terms = [t for t in [name, email] if t]
            if not search_terms:
                return []

            # Search sender and subject fields
            for term in search_terms:
                rows = store.conn.execute(
                    """SELECT id, node_type, source, timestamp, subject, sender,
                              body_preview, data
                       FROM nodes
                       WHERE node_type = 'email'
                         AND (LOWER(sender) LIKE ? OR LOWER(subject) LIKE ?
                              OR LOWER(recipients) LIKE ? OR LOWER(body_preview) LIKE ?)
                       ORDER BY timestamp DESC
                       LIMIT 10""",
                    (f"%{term}%", f"%{term}%", f"%{term}%", f"%{term}%"),
                ).fetchall()
                for r in rows:
                    node = dict(r)
                    ts_str = ""
                    if node.get("timestamp"):
                        try:
                            dt = datetime.fromtimestamp(node["timestamp"], tz=local_tz)
                            ts_str = dt.strftime("%Y-%m-%d %H:%M")
                        except (OSError, ValueError):
                            pass
                    results.append({
                        "id": node["id"],
                        "subject": node.get("subject", ""),
                        "sender": node.get("sender", ""),
                        "date": ts_str,
                        "body_preview": (node.get("body_preview") or "")[:400],
                        "query_source": f"person_emails:{term}",
                    })

        elif qtype == "person_messages":
            name = (params.get("name") or "").lower()
            if not name:
                return []
            rows = store.conn.execute(
                """SELECT id, node_type, source, timestamp, subject, sender,
                          body_preview, data
                   FROM nodes
                   WHERE node_type = 'message'
                     AND (LOWER(sender) LIKE ? OR LOWER(body_preview) LIKE ?
                          OR LOWER(subject) LIKE ?)
                   ORDER BY timestamp DESC
                   LIMIT 8""",
                (f"%{name}%", f"%{name}%", f"%{name}%"),
            ).fetchall()
            for r in rows:
                node = dict(r)
                ts_str = ""
                if node.get("timestamp"):
                    try:
                        dt = datetime.fromtimestamp(node["timestamp"], tz=local_tz)
                        ts_str = dt.strftime("%Y-%m-%d %H:%M")
                    except (OSError, ValueError):
                        pass
                results.append({
                    "id": node["id"],
                    "subject": node.get("subject", ""),
                    "sender": node.get("sender", ""),
                    "date": ts_str,
                    "body_preview": (node.get("body_preview") or "")[:400],
                    "query_source": f"person_messages:{name}",
                })

        elif qtype == "topic_search":
            keywords = params.get("keywords", [])
            if not keywords:
                return []
            for kw in keywords[:3]:
                kw_lower = kw.lower()
                rows = store.conn.execute(
                    """SELECT id, node_type, source, timestamp, subject, sender,
                              body_preview, data
                       FROM nodes
                       WHERE (LOWER(subject) LIKE ? OR LOWER(body_preview) LIKE ?)
                         AND node_type IN ('email', 'message', 'meeting')
                       ORDER BY timestamp DESC
                       LIMIT 5""",
                    (f"%{kw_lower}%", f"%{kw_lower}%"),
                ).fetchall()
                for r in rows:
                    node = dict(r)
                    ts_str = ""
                    if node.get("timestamp"):
                        try:
                            dt = datetime.fromtimestamp(node["timestamp"], tz=local_tz)
                            ts_str = dt.strftime("%Y-%m-%d %H:%M")
                        except (OSError, ValueError):
                            pass
                    results.append({
                        "id": node["id"],
                        "subject": node.get("subject", ""),
                        "sender": node.get("sender", ""),
                        "date": ts_str,
                        "body_preview": (node.get("body_preview") or "")[:400],
                        "query_source": f"topic_search:{kw}",
                    })

        elif qtype == "recent_from_person":
            name = (params.get("name") or "").lower()
            days = params.get("days", 14)
            if not name:
                return []
            cutoff = int((datetime.now() - timedelta(days=days)).timestamp())
            rows = store.conn.execute(
                """SELECT id, node_type, source, timestamp, subject, sender,
                          body_preview, data
                   FROM nodes
                   WHERE (LOWER(sender) LIKE ? OR LOWER(recipients) LIKE ?
                          OR LOWER(subject) LIKE ? OR LOWER(body_preview) LIKE ?)
                     AND timestamp >= ?
                     AND node_type IN ('email', 'message', 'meeting')
                   ORDER BY timestamp DESC
                   LIMIT 10""",
                (f"%{name}%", f"%{name}%", f"%{name}%", f"%{name}%", cutoff),
            ).fetchall()
            for r in rows:
                node = dict(r)
                ts_str = ""
                if node.get("timestamp"):
                    try:
                        dt = datetime.fromtimestamp(node["timestamp"], tz=local_tz)
                        ts_str = dt.strftime("%Y-%m-%d %H:%M")
                    except (OSError, ValueError):
                        pass
                results.append({
                    "id": node["id"],
                    "subject": node.get("subject", ""),
                    "sender": node.get("sender", ""),
                    "date": ts_str,
                    "body_preview": (node.get("body_preview") or "")[:400],
                    "query_source": f"recent_from_person:{name}",
                })

        elif qtype == "commitments_search":
            keywords = params.get("keywords", [])
            if not keywords:
                return []
            all_open = derived.conn.execute(
                "SELECT * FROM commitments WHERE status = 'open'"
            ).fetchall()
            for c in all_open:
                c = dict(c)
                c_text = f"{c.get('who', '')} {c.get('to_whom', '')} {c.get('what', '')} {c.get('note', '')}".lower()
                for kw in keywords:
                    if kw.lower() in c_text:
                        results.append({
                            "id": c["id"],
                            "type": c["commitment_type"],
                            "what": c["what"],
                            "who": c.get("who", ""),
                            "to_whom": c.get("to_whom", ""),
                            "status": c["status"],
                            "deadline": c.get("deadline"),
                            "priority": c["priority"],
                            "note": (c.get("note") or "")[:250],
                            "query_source": f"commitments_search:{kw}",
                        })
                        break  # don't double-add

        elif qtype == "thread_lookup":
            thread_id = params.get("thread_id", "")
            if not thread_id:
                return []
            rows = store.conn.execute(
                """SELECT id, node_type, source, timestamp, subject, sender,
                          body_preview, data
                   FROM nodes
                   WHERE thread_id = ?
                   ORDER BY timestamp ASC""",
                (thread_id,),
            ).fetchall()
            for r in rows:
                node = dict(r)
                ts_str = ""
                if node.get("timestamp"):
                    try:
                        dt = datetime.fromtimestamp(node["timestamp"], tz=local_tz)
                        ts_str = dt.strftime("%Y-%m-%d %H:%M")
                    except (OSError, ValueError):
                        pass
                results.append({
                    "id": node["id"],
                    "subject": node.get("subject", ""),
                    "sender": node.get("sender", ""),
                    "date": ts_str,
                    "body_preview": (node.get("body_preview") or "")[:400],
                    "query_source": f"thread_lookup:{thread_id}",
                })

        elif qtype == "meeting_lookup":
            # Load full meeting notes by node ID
            meeting_id = params.get("id", "")
            if not meeting_id:
                return []
            row = store.conn.execute(
                """SELECT id, node_type, source, timestamp, subject, sender,
                          body_preview, data
                   FROM nodes
                   WHERE id = ? AND node_type = 'meeting'""",
                (meeting_id,),
            ).fetchone()
            if row:
                node = dict(row)
                ts_str = ""
                if node.get("timestamp"):
                    try:
                        dt = datetime.fromtimestamp(node["timestamp"], tz=local_tz)
                        ts_str = dt.strftime("%Y-%m-%d %H:%M")
                    except (OSError, ValueError):
                        pass
                meeting_body = ""
                if node.get("data"):
                    try:
                        mdata = msgpack.unpackb(node["data"], raw=False)
                        meeting_body = mdata.get("body", "") or ""
                    except Exception:
                        pass
                if not meeting_body:
                    meeting_body = node.get("body_preview", "") or ""
                results.append({
                    "id": node["id"],
                    "subject": node.get("subject", ""),
                    "sender": node.get("sender", ""),
                    "date": ts_str,
                    "body_preview": (node.get("body_preview") or "")[:400],
                    "notes_preview": meeting_body[:1500],
                    "query_source": f"meeting_lookup:{meeting_id}",
                })

    except Exception as e:
        logger.warning("Graph query failed (%s): %s", qtype, e)

    # Deduplicate by ID
    seen_ids = set()
    deduped = []
    for r in results:
        if r["id"] not in seen_ids:
            seen_ids.add(r["id"])
            deduped.append(r)
    return deduped


def _triage_meeting_context(
    event: dict,
    context: dict,
    llm: LLMClient,
) -> dict:
    """Pass 1: Ask the LLM to assess context quality and request missing info.

    Returns the triage response dict with:
      - relevant_ids: IDs the LLM considers relevant
      - irrelevant_ids: IDs to drop
      - missing: list of graph queries to execute
      - assessment: 1-2 sentence summary
    """
    formatted = _format_meeting_context_for_llm(event, context)

    user_message = (
        f"Review the following context for an upcoming meeting and assess its quality.\n\n"
        f"{formatted}"
    )

    result = llm.run_json(TRIAGE_SYSTEM_PROMPT, user_message)

    # Validate structure
    if not isinstance(result, dict):
        logger.warning("Triage returned non-dict: %s", type(result))
        return {"relevant_ids": [], "irrelevant_ids": [], "missing": [], "assessment": ""}

    return {
        "relevant_ids": result.get("relevant_ids", []),
        "irrelevant_ids": result.get("irrelevant_ids", []),
        "stale_ids": result.get("stale_ids", []),
        "detected_shift": result.get("detected_shift", {"found": False}),
        "missing": result.get("missing", [])[:5],  # cap at 5
        "assessment": result.get("assessment", ""),
    }


def _refine_context(
    context: dict,
    triage: dict,
    additional_results: dict[str, list[dict]],
) -> dict:
    """Apply triage feedback: drop irrelevant items, merge new results, mark stale.

    Returns a new context dict with refined content.
    """
    irrelevant = set(triage.get("irrelevant_ids", []))
    stale = set(triage.get("stale_ids", []))

    # Filter existing context (drop irrelevant, mark stale)
    def _mark_stale(items, exclude_ids):
        result = []
        for item in items:
            if item["id"] in exclude_ids:
                continue
            if item["id"] in stale:
                item["_stale"] = True
            result.append(item)
        return result

    refined = {
        "related_emails": _mark_stale(context["related_emails"], irrelevant),
        "related_meetings": _mark_stale(context["related_meetings"], irrelevant),
        "related_messages": _mark_stale(context["related_messages"], irrelevant),
        "commitments": context["commitments"],  # don't filter commitments
        "contact_nodes": context.get("contact_nodes", []),
    }

    # Carry through shift detection for the synthesis prompt
    detected_shift = triage.get("detected_shift", {"found": False})
    if detected_shift.get("found"):
        refined["_detected_shift"] = detected_shift

    # Merge additional results from graph queries
    existing_ids = set()
    for lst in [refined["related_emails"], refined["related_meetings"], refined["related_messages"]]:
        for item in lst:
            existing_ids.add(item["id"])

    for query_key, results in additional_results.items():
        for r in results:
            if r["id"] in existing_ids:
                continue
            existing_ids.add(r["id"])

            # Route to appropriate category
            if "commitments_search" in r.get("query_source", ""):
                # Add to commitments if not already present
                existing_commitment_ids = {c.get("id") for c in refined["commitments"] if c.get("id")}
                if r["id"] not in existing_commitment_ids:
                    refined["commitments"].append(r)
            elif any(kw in r.get("query_source", "") for kw in ["person_messages", "message"]):
                refined["related_messages"].append(r)
            else:
                # Default: treat as email/general content
                refined["related_emails"].append(r)

    # Re-sort by date
    for lst in [refined["related_emails"], refined["related_meetings"], refined["related_messages"]]:
        lst.sort(key=lambda x: x.get("date", ""), reverse=True)

    # Update summary
    parts = []
    if refined["related_emails"]:
        parts.append(f"{len(refined['related_emails'])} related emails")
    if refined["related_meetings"]:
        parts.append(f"{len(refined['related_meetings'])} past meetings")
    if refined["related_messages"]:
        parts.append(f"{len(refined['related_messages'])} message threads")
    if refined["commitments"]:
        parts.append(f"{len(refined['commitments'])} commitments")
    refined["context_summary"] = ", ".join(parts) if parts else "no context"

    return refined


def _agentic_context_refinement(
    event: dict,
    context: dict,
    store: GraphStore,
    derived: DerivedStore,
    llm: LLMClient,
    user_tz: str = "America/Los_Angeles",
) -> dict:
    """Two-pass agentic refinement for a single meeting's context.

    Pass 1: Send raw context to LLM for triage assessment.
    Refinement: Execute requested graph queries.
    Returns: Refined context dict ready for final synthesis.
    """
    # Pass 1: Triage
    console.print("      [dim]Pass 1: Triage assessment...[/dim]")
    triage = _triage_meeting_context(event, context, llm)

    assessment = triage.get("assessment", "")
    if assessment:
        console.print(f"      [italic]{assessment}[/italic]")

    # Log shift detection
    detected_shift = triage.get("detected_shift", {})
    if detected_shift.get("found"):
        console.print(f"      [bold yellow]⚡ Shift detected:[/bold yellow] {detected_shift.get('description', '?')}")
        stale_count = len(triage.get("stale_ids", []))
        if stale_count:
            console.print(f"      [yellow]{stale_count} items marked as pre-shift (stale)[/yellow]")

    missing = triage.get("missing", [])
    if not missing:
        console.print("      [dim]Context sufficient — no additional queries needed[/dim]")
        # Still apply irrelevant filtering
        return _refine_context(context, triage, {})

    # Execute graph queries
    console.print(f"      [bold]Executing {len(missing)} graph queries...[/bold]")
    additional_results = {}
    for i, query in enumerate(missing):
        qtype = query.get("query_type", "?")
        reason = query.get("reason", "")
        console.print(f"        [{i+1}] {qtype}: {reason[:80]}")

        results = _execute_graph_query(query, store, derived, user_tz=user_tz)
        if results:
            console.print(f"            → {len(results)} results")
            additional_results[f"{qtype}_{i}"] = results
        else:
            console.print("            → no results")

    # Apply refinement
    refined = _refine_context(context, triage, additional_results)
    console.print(f"      [green]Refined: {refined['context_summary']}[/green]")

    return refined


# ══════════════════════════════════════════════════════════════════
# Briefing synthesis — format context for the LLM
# ══════════════════════════════════════════════════════════════════

def _format_meeting_context_for_llm(
    event: dict,
    context: dict,
    web_results: list[dict] | None = None,
) -> str:
    """Format all gathered context into a structured prompt for the synthesizer."""
    parts = []

    # Meeting metadata
    parts.append(f"# Meeting: {event.get('subject', '(untitled)')}")
    parts.append(f"**Classification:** {event.get('_classification', 'unknown')}")
    parts.append(f"**When:** {event.get('_local_time', '(unknown)')}")

    attendee_str = ", ".join(
        f"{a['name']} ({a['email']})" if a['email'] else a['name']
        for a in event["_attendees"]
    ) or "(no attendees listed)"
    parts.append(f"**Attendees:** {attendee_str}")

    data = event.get("_data", {})
    if data.get("location"):
        parts.append(f"**Location:** {data['location']}")
    if data.get("event_url"):
        parts.append(f"**Link:** {data['event_url']}")
    if data.get("duration_minutes"):
        parts.append(f"**Duration:** {data['duration_minutes']} minutes")

    # RSVP status of attendees (if available)
    if data.get("attendees"):
        rsvp_info = []
        for att in data["attendees"]:
            if isinstance(att, dict) and att.get("response_status"):
                name = att.get("name") or att.get("email", "?")
                rsvp_info.append(f"{name}: {att['response_status']}")
        if rsvp_info:
            parts.append(f"**RSVP status:** {', '.join(rsvp_info)}")

    # Event body/description
    body = data.get("body", "") or event.get("body_preview", "") or ""
    if body.strip():
        parts.append(f"\n**Event description:**\n{body[:600]}")

    # ── Strategic shift detection ──
    # If triage detected a pivot/direction change, flag it prominently
    detected_shift = context.get("_detected_shift")
    if detected_shift and detected_shift.get("found"):
        parts.append("\n## ⚠️ STRATEGIC SHIFT DETECTED")
        parts.append(f"**{detected_shift.get('description', 'Direction changed recently.')}**")
        if detected_shift.get("pivot_date_approx"):
            parts.append(f"Approximate date: {detected_shift['pivot_date_approx']}")
        parts.append("Items below marked [STALE] predate this shift and may no longer "
                     "reflect current priorities. Lead with post-shift context.")

    # Contact dossiers from attendee edges
    if context.get("contact_nodes"):
        parts.append("\n## Attendee Dossiers")
        for contact in context["contact_nodes"]:
            cdata = contact.get("_data", {})
            name = cdata.get("display_name", contact["id"])
            email = cdata.get("email", "")
            parts.append(f"- **{name}** ({email})")

    # Related emails (sorted by recency)
    if context["related_emails"]:
        parts.append("\n## Related Emails (newest first)")
        for e in context["related_emails"]:
            hop_tag = f" [hop-{e['hop']}]" if e.get("hop", 0) > 1 else ""
            stale_tag = " [STALE]" if e.get("_stale") else ""
            rel_tag = f" [rel={e['relevance']}]" if e.get("relevance") else ""
            parts.append(f"- [{e['date']}]{stale_tag}{rel_tag} From: {e['sender']} | Subject: {e['subject']}{hop_tag}")
            if e.get("body_preview"):
                parts.append(f"  Preview: {e['body_preview'][:250]}")

    # Past meetings
    if context["related_meetings"]:
        parts.append("\n## Past Meetings / Calendar Events with These People")
        for m in context["related_meetings"]:
            stale_tag = " [STALE]" if m.get("_stale") else ""
            rel_tag = f" [relevance={m['relevance']}]" if m.get("relevance") else ""
            parts.append(f"- [{m['date']}]{stale_tag}{rel_tag} {m['subject']}")
            if m.get("notes_preview"):
                parts.append(f"  Notes: {m['notes_preview'][:400]}")

    # Related messages (iMessage, Slack)
    if context["related_messages"]:
        parts.append("\n## Related Messages (iMessage/Slack)")
        for msg in context["related_messages"]:
            stale_tag = " [STALE]" if msg.get("_stale") else ""
            parts.append(f"- [{msg['date']}]{stale_tag} {msg['sender']}: {msg['body_preview'][:200]}")

    # Meeting manifest (older meetings available on request)
    if context.get("meeting_manifest"):
        parts.append("\n## Additional Past Meetings (available via meeting_lookup query)")
        parts.append("Request any of these by ID if relevant to the briefing:")
        for m in context["meeting_manifest"]:
            rel_tag = f" [rel={m['relevance']}]" if m.get("relevance") else ""
            parts.append(f"- [{m['date']}]{rel_tag} {m['subject']} (id: {m['id']})")

    # Open commitments
    if context["commitments"]:
        parts.append("\n## Open Commitments Involving Attendees or Topic")
        for c in context["commitments"]:
            deadline = f" (due {c['deadline']})" if c.get("deadline") else ""
            overdue = ""
            if c.get("deadline"):
                try:
                    dl = datetime.strptime(c["deadline"], "%Y-%m-%d")
                    if dl < datetime.now():
                        overdue = " OVERDUE"
                except (ValueError, TypeError):
                    pass
            parts.append(
                f"- P{c['priority']} [{c['type']}] {c['what']}{deadline}{overdue}"
                f"\n  Who: {c.get('who', '?')} -> {c.get('to_whom', '?')}"
            )
            if c.get("note"):
                parts.append(f"  Context: {c['note'][:200]}")

    # Web search results
    if web_results:
        parts.append("\n## Web Search Results")
        for wr in web_results:
            parts.append(f"\n**Query:** {wr['query']}")
            for r in wr.get("results", []):
                parts.append(f"- {r['title']}: {r['snippet'][:200]}")
                parts.append(f"  URL: {r['url']}")

    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════
# User questions — things only the human can answer
# ══════════════════════════════════════════════════════════════════

def _generate_user_questions(
    events_with_context: list[tuple[dict, dict, list[dict] | None]],
    llm: LLMClient,
) -> list[dict]:
    """After triage and refinement, ask the LLM what questions it has for the user.

    Returns a list of question dicts with:
      - event_subject: which event this question relates to
      - question: the question text
      - category: LOGISTICS | DECISIONS | MISSING_CONTEXT | MATERIALS | COORDINATION
      - context_if_answered: what the answer would improve
      - accepts_file: bool — whether the user can provide a file path
    """
    parts = []
    for event, context, web_results in events_with_context:
        if event.get("_classification") == "holiday":
            continue
        formatted = _format_meeting_context_for_llm(event, context, web_results)
        parts.append(formatted)

    if not parts:
        return []

    separator = "\n\n" + "-" * 40 + "\n\n"
    user_message = (
        f"Review the following {len(parts)} events and their context. "
        f"What questions would you ask the user to improve the briefing?\n\n"
        + separator.join(parts)
    )

    result = llm.run_json(QUESTIONS_SYSTEM_PROMPT, user_message)

    if isinstance(result, dict) and "questions" in result:
        questions = result["questions"]
        if isinstance(questions, list):
            return questions[:5]  # safety cap — prompt should self-limit to 1-3
    return []


def _present_questions_cli(questions: list[dict]) -> dict[int, str]:
    """Present questions to the user via CLI and collect responses.

    Returns a dict of {question_index: response_text}.
    Empty string means user skipped that question.
    """
    from rich.panel import Panel

    if not questions:
        return {}

    console.print()
    console.rule("[bold yellow]Questions for you[/bold yellow]")
    console.print()
    console.print(
        "[dim]Answer any questions that would help improve your briefing.\n"
        "Press Enter to skip a question. Type a file path to attach a document.\n"
        "Type 'skip' to skip all remaining questions.[/dim]\n"
    )

    responses = {}
    skip_all = False

    for i, q in enumerate(questions):
        if skip_all:
            break

        cat = q.get("category", "")
        cat_colors = {
            "LOGISTICS": "blue",
            "DECISIONS": "magenta",
            "MISSING_CONTEXT": "yellow",
            "MATERIALS": "cyan",
            "COORDINATION": "green",
        }
        cat_color = cat_colors.get(cat, "white")

        event_subj = q.get("event_subject", "General")
        question_text = q.get("question", "")
        file_hint = " [dim](file path OK)[/dim]" if q.get("accepts_file") else ""

        console.print(
            f"  [{cat_color}][{cat}][/{cat_color}] "
            f"[bold]{event_subj}[/bold]"
        )
        console.print(f"  {question_text}{file_hint}")

        try:
            response = console.input("  [dim]\u2192[/dim] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n  [dim]Skipping remaining questions[/dim]")
            break

        if response.lower() in ("skip", "s"):
            skip_all = True
            continue

        if response:
            responses[i] = response

    console.print()
    return responses


def _extract_pdf_text(path) -> str:
    """Extract text from a PDF with multiple fallback methods.

    Tries in order: pdftotext (poppler), PyPDF2, pymupdf (fitz).
    Returns extracted text or a failure message starting with '('.
    """
    import subprocess

    # Method 1: pdftotext (poppler) — best quality for text PDFs
    try:
        result = subprocess.run(
            ["pdftotext", str(path), "-"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout[:8000]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    except Exception:
        pass

    # Method 2: PyPDF2 (pure Python, commonly installed)
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(str(path))
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        if text_parts:
            return "\n".join(text_parts)[:8000]
    except ImportError:
        pass
    except Exception:
        pass

    # Method 3: pymupdf / fitz
    try:
        import fitz
        doc = fitz.open(str(path))
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        if text_parts:
            return "\n".join(text_parts)[:8000]
    except ImportError:
        pass
    except Exception:
        pass

    return f"(PDF at {path}, could not extract text — install poppler, PyPDF2, or pymupdf)"


def _process_user_responses(
    questions: list[dict],
    responses: dict[int, str],
) -> str:
    """Convert user responses into additional context for the synthesis prompt.

    If a response looks like a file path and the question accepts files, reads the file.
    Returns a markdown string to inject into the synthesis.
    """
    if not responses:
        return ""

    parts = ["## User-Provided Context"]

    for idx, response in sorted(responses.items()):
        if idx >= len(questions):
            continue

        q = questions[idx]
        event_subj = q.get("event_subject", "General")
        question_text = q.get("question", "")

        # Check if response is a file path (only if the question accepts files)
        file_content = None
        if q.get("accepts_file") and response:
            # Heuristic: only try as a file path if it looks like one.
            # File paths typically: start with / or ~ or ., contain a dot extension,
            # have no spaces (or are quoted), and are short.
            # Prose answers: contain spaces, are longer, start with words.
            response_stripped = response.strip().strip("'\"")
            looks_like_path = (
                response_stripped.startswith(("/", "~", "."))
                or (len(response_stripped.split()) == 1 and "." in response_stripped)
            )
            if looks_like_path:
                p = Path(response_stripped).expanduser()
                if p.exists() and p.is_file():
                    try:
                        if p.suffix in (".txt", ".md", ".csv", ".json", ".py", ".html"):
                            file_content = p.read_text()[:8000]
                            console.print(f"    [green]Read {len(file_content)} chars from {p.name}[/green]")
                        elif p.suffix == ".pdf":
                            file_content = _extract_pdf_text(p)
                            if file_content and not file_content.startswith("("):
                                console.print(f"    [green]Extracted {len(file_content)} chars from {p.name}[/green]")
                            else:
                                console.print(f"    [yellow]PDF extraction limited: {p.name}[/yellow]")
                        elif p.suffix == ".docx":
                            try:
                                import subprocess
                                result = subprocess.run(
                                    ["pandoc", str(p), "-t", "plain"],
                                    capture_output=True, text=True, timeout=10,
                                )
                                if result.returncode == 0:
                                    file_content = result.stdout[:8000]
                                    console.print(f"    [green]Extracted {len(file_content)} chars from {p.name}[/green]")
                            except Exception:
                                file_content = f"(DOCX at {p}, could not extract text)"
                        else:
                            file_content = f"(File at {p}, type {p.suffix})"
                    except Exception as e:
                        file_content = f"(Could not read file: {e})"
                elif not p.exists():
                    console.print(f"    [yellow]File not found: {p}[/yellow]")

        parts.append(f"\n### Re: {event_subj}")
        parts.append(f"**Q:** {question_text}")
        if file_content and not file_content.startswith("("):
            parts.append(f"**A:** User provided document: {Path(response).expanduser().name}")
            parts.append(
                f"**IMPORTANT — Analyze this document thoroughly.** Extract key skills, "
                f"experience, gaps, and anything relevant to the event. Do not just mention "
                f"the filename — use the content to generate specific, actionable insights."
            )
            parts.append(f"```\n{file_content}\n```")
        elif file_content:
            # Extraction failed partially
            parts.append(f"**A:** {file_content}")
        else:
            parts.append(f"**A:** {response}")

    return "\n".join(parts)


def _gather_ambient_context(
    events: list[dict],
    store: GraphStore | None = None,
    web_search: bool = False,
    user_tz: str = "America/Los_Angeles",
) -> str:
    """Gather ambient context from the user profile and event logistics.

    Reads ~/.alteris/profile.yaml for user-specific data (home, family, preferences).
    Falls back gracefully if no profile exists.

    Returns a markdown string to prepend to the synthesis prompt.
    """
    import zoneinfo

    local_tz = zoneinfo.ZoneInfo(user_tz)
    now = datetime.now(local_tz)
    profile = _load_user_profile()

    parts = []

    # ── User identity ──
    parts.append("## User Profile")
    if profile.get("name"):
        parts.append(f"- Name: {profile['name']}")
    home = profile.get("home", {})
    if home:
        home_str = ", ".join(filter(None, [
            home.get("neighborhood"), home.get("city"), home.get("state"),
        ]))
        if home_str:
            parts.append(f"- Home: {home_str}")
    if profile.get("role"):
        parts.append(f"- Role: {profile['role']}")
    if profile.get("context"):
        parts.append(f"- Current situation: {profile['context']}")

    # ── Family & dependents ──
    family = profile.get("family", {})
    if family:
        parts.append("\n## Family & Dependents")
        if family.get("spouse"):
            parts.append(f"- Spouse/partner: {family['spouse']}")
        for child in family.get("children", []):
            details = child.get("name", "Child")
            if child.get("school"):
                details += f" (attends {child['school']})"
            elif child.get("daycare"):
                details += f" (attends {child['daycare']})"
            parts.append(f"- Child: {details}")
        for provider in family.get("care_providers", []):
            parts.append(f"- Care provider: {provider.get('name', '?')} ({provider.get('role', '')})")
        for pet in family.get("pets", []):
            details = f"{pet.get('name', 'Pet')} ({pet.get('type', 'pet')})"
            if pet.get("daycare"):
                details += f" — daycare: {pet['daycare']}"
            if pet.get("walker"):
                details += f" — walker: {pet['walker']}"
            parts.append(f"- Pet: {details}")

        parts.append("- IMPORTANT: When you see a holiday, closure, or schedule disruption "
                     "in the context, cross-reference with dependents listed above. If a "
                     "closure affects them and no arrangements are mentioned, flag it and "
                     "draft a message to a care provider.")

    # ── Travel preferences ──
    travel = profile.get("travel", {})
    if travel:
        parts.append("\n## Travel Preferences")
        if travel.get("airline_loyalty"):
            parts.append(f"- Airline loyalty: {travel['airline_loyalty']}")
        if travel.get("seat_preference"):
            parts.append(f"- Seat preference: {travel['seat_preference']}")
        if travel.get("class_preference"):
            parts.append(f"- Class preference: {travel['class_preference']}")
        if travel.get("hotel_loyalty"):
            parts.append(f"- Hotel loyalty: {travel['hotel_loyalty']}")
        if travel.get("rental_car"):
            parts.append(f"- Rental car: {travel['rental_car']}")

    # ── Local knowledge ──
    local_knowledge = profile.get("local_knowledge", [])
    if local_knowledge:
        city = home.get("city", "")
        neighborhood = home.get("neighborhood", "")
        header = f"Local Knowledge ({neighborhood}, {city})" if neighborhood else f"Local Knowledge ({city})" if city else "Local Knowledge"
        parts.append(f"\n## {header}")
        for item in local_knowledge:
            parts.append(f"- {item}")

    # General recommendation principle (always applies)
    parts.append("\n## Recommendation Principles")
    parts.append("- When recommending anything (gifts, flowers, food, services), tier by budget: affordable → mid → premium")
    parts.append("- Prefer local options the user can reach easily over generic online suggestions")

    # ── Day awareness ──
    parts.append(f"\n## Today")
    parts.append(f"- Date: {now.strftime('%A, %B %d, %Y')}")
    parts.append(f"- Time now: {now.strftime('%I:%M %p %Z')}")

    # ── Event locations ──
    event_locations = set()
    for e in events:
        data = e.get("_data", {})
        if data.get("location"):
            event_locations.add(data["location"])

    if event_locations:
        parts.append(f"\n## Event Locations")
        for loc in event_locations:
            parts.append(f"- {loc}")
        if home:
            parts.append(f"- User lives in {home.get('neighborhood', home.get('city', 'their home area'))}. "
                         "Consider travel times from there.")

    # ── Weather ──
    if web_search:
        city = home.get("city", "the user's area")
        parts.append("\n## Weather")
        parts.append(f"(See web search results for current {city} weather)")

    return "\n".join(parts)


def _synthesize_briefing(
    events_with_context: list[tuple[dict, dict, list[dict] | None]],
    llm: LLMClient,
    ambient_context: str = "",
    user_context: str = "",
) -> str:
    """Send all meeting contexts to the LLM and get back a unified briefing."""
    all_parts = []
    for event, context, web_results in events_with_context:
        formatted = _format_meeting_context_for_llm(event, context, web_results)
        all_parts.append(formatted)

    separator = "\n\n" + "=" * 60 + "\n\n"
    events_block = separator.join(all_parts)

    event_count = len(events_with_context)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M %Z")
    header = (
        f"Current time: {now_str}\n"
        f"Generate briefings for {event_count} upcoming event{'s' if event_count > 1 else ''}.\n"
        f"For each event, think: what does the user need to succeed here?\n"
        f"Holidays -> single line. Everything else -> BLUF + relevant sections only.\n"
        f"End with Cross-Event Notes if connections or logistics conflicts exist.\n"
    )

    user_message = header
    if ambient_context:
        user_message += f"\n{ambient_context}\n\n"
    if user_context:
        user_message += f"\n{user_context}\n\n"
    user_message += events_block

    return llm.run(BRIEFING_SYSTEM_PROMPT, user_message)


# ══════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════

def run_briefing(
    store: GraphStore,
    derived: DerivedStore,
    days_ahead: int = 7,
    web_search: bool = False,
    interactive: bool = False,
    user_tz: str = "America/Los_Angeles",
    model: str = "gemini-3-flash-preview",
    thinking: str = "medium",
) -> dict:
    """Run the meeting briefing pipeline.

    Architecture (agentic, optionally interactive):
      1. Discover upcoming events, classify, dedup
      2. Gather raw context via multi-hop graph walk (generous, may be noisy)
      3. For professional meetings: agentic triage pass
         - LLM reviews raw context, marks relevant/irrelevant, requests missing data
         - Execute graph queries for missing data
         - Refine context (drop irrelevant, merge new results)
      4. Optional web search for external meetings
      5. Interactive: ask user questions the graph can't answer
      6. Gather ambient context (user profile, logistics)
      7. Final synthesis with all context merged

    Args:
        store: Graph store (source data)
        derived: Derived store (commitments)
        days_ahead: How many days ahead to look
        web_search: Whether to run web searches for context
        interactive: Whether to ask user questions before synthesis
        user_tz: Timezone for display
        model: LLM model for synthesis
        thinking: Thinking level for LLM

    Returns:
        dict with 'briefing' (markdown), 'events_count', 'model'
    """
    t0 = time.time()
    console.print(f"\n[bold]Meeting Briefing[/bold] (next {days_ahead} days, web_search={'on' if web_search else 'off'})")

    # Step 1: Find upcoming events (with classification and dedup)
    events = _get_upcoming_events(store, days_ahead=days_ahead, user_tz=user_tz)
    if not events:
        console.print("  [yellow]No upcoming calendar events found.[/yellow]")
        return {"briefing": "No upcoming meetings found.", "events_count": 0, "model": model}

    console.print(f"  Found {len(events)} upcoming events:")
    for e in events:
        tag = {
            "holiday": "\U0001f3d6\ufe0f ",
            "personal": "\U0001f389",
            "personal_external": "\U0001f381",
            "professional_internal": "\U0001f3e2",
            "professional_external": "\U0001f91d",
        }.get(e["_classification"], "\U0001f4c5")
        console.print(f"    {tag} {e['_local_time']}  {e.get('subject', '(untitled)')}  [{e['_classification']}]")

    # Step 2: Gather raw context per event
    llm = LLMClient(provider="gemini", model=model, thinking_level=thinking)
    events_with_context = []

    for event in events:
        classification = event["_classification"]

        # Holidays: search for anything related to this specific holiday in ±7 days.
        # Don't assume what matters — let the graph tell us. Could be school closures,
        # travel bookings, pet boarding, event tickets, anything.
        if classification == "holiday":
            holiday_name = (event.get("subject") or "").strip()
            console.print(f"\n  [dim]Holiday:[/dim] {holiday_name}")

            holiday_context = {
                "related_emails": [], "related_meetings": [],
                "related_messages": [], "meeting_manifest": [],
                "commitments": [],
                "contact_nodes": [], "context_summary": "holiday",
            }

            event_ts = event.get("timestamp", 0)
            window_start = event_ts - 10 * 86400  # 10 days before
            window_end = event_ts + 2 * 86400     # 2 days after

            # Extract search terms from the holiday name itself + generic closure terms
            # "Presidents' Day" → ["presidents", "president"]
            # "Thanksgiving" → ["thanksgiving"]
            import re
            name_words = re.findall(r"[a-zA-Z]{4,}", holiday_name.lower())
            # Add generic terms that signal something is affected by *any* holiday
            search_terms = list(set(name_words)) + ["closed", "closure", "holiday"]

            seen_ids = set()
            for term in search_terms[:5]:
                rows = store.conn.execute(
                    """SELECT id, node_type, source, timestamp, subject, sender,
                              body_preview, data
                       FROM nodes
                       WHERE (LOWER(subject) LIKE ? OR LOWER(body_preview) LIKE ?)
                         AND node_type IN ('email', 'message')
                         AND timestamp >= ? AND timestamp <= ?
                       ORDER BY timestamp DESC
                       LIMIT 5""",
                    (f"%{term}%", f"%{term}%", window_start, window_end),
                ).fetchall()
                for r in rows:
                    node = dict(r)
                    if node["id"] not in seen_ids:
                        seen_ids.add(node["id"])
                        # Normalize to match the format _gather_meeting_context produces
                        ts = node.get("timestamp", 0)
                        date_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else ""
                        normalized = {
                            "id": node["id"],
                            "date": date_str,
                            "sender": node.get("sender", ""),
                            "subject": node.get("subject", ""),
                            "body_preview": node.get("body_preview", ""),
                            "hop": 0,
                        }
                        holiday_context["related_emails"].append(normalized)

            if holiday_context["related_emails"]:
                n = len(holiday_context["related_emails"])
                holiday_context["context_summary"] = f"holiday — {n} related messages found"
                console.print(f"    Found {n} related emails/messages")

            events_with_context.append((event, holiday_context, None))
            continue

        console.print(f"\n  [bold]Gathering context:[/bold] {event.get('subject', '(untitled)')}")

        # Gather raw context (generous caps — triage will refine)
        if classification in ("personal", "personal_external"):
            context = _gather_meeting_context(
                event, store, derived,
                max_emails=6, max_meetings=3, max_messages=4,
                user_tz=user_tz,
            )
        else:
            context = _gather_meeting_context(
                event, store, derived,
                max_emails=12, max_meetings=6, max_messages=8,
                user_tz=user_tz,
            )
        console.print(f"    Raw: {context['context_summary']}")

        # Step 3: Agentic triage for professional meetings
        if classification in ("professional_external", "professional_internal"):
            context = _agentic_context_refinement(
                event, context, store, derived, llm, user_tz=user_tz,
            )

        # Step 4: Web search
        # - professional_external: attendee/company research
        # - personal_external: gift ideas based on event context
        web_results = None
        needs_web_search = (
            web_search and classification in ("professional_external", "personal_external")
        )
        if needs_web_search:
            if classification == "personal_external":
                # Generate gift-oriented search queries
                queries = _generate_gift_search_queries(event)
            else:
                queries = _generate_search_queries(event, context, llm)

            if queries:
                console.print(f"    \U0001f50d Searching: {', '.join(queries)}")
                web_results = _web_search(queries)
                if web_results:
                    total_hits = sum(len(wr.get("results", [])) for wr in web_results)
                    console.print(f"    Found {total_hits} web results")
            else:
                console.print("    [dim]No web search needed[/dim]")

        events_with_context.append((event, context, web_results))

    # Step 5: Interactive questions (if enabled)
    user_context = ""
    if interactive:
        console.print(f"\n  [bold]Generating questions...[/bold]")
        questions = _generate_user_questions(events_with_context, llm)
        if questions:
            responses = _present_questions_cli(questions)
            if responses:
                user_context = _process_user_responses(questions, responses)
                console.print(f"  [green]Incorporated {len(responses)} response(s) into briefing[/green]")
            else:
                console.print("  [dim]No responses provided — proceeding with available context[/dim]")
        else:
            console.print("  [dim]No questions needed — context looks sufficient[/dim]")

    # Step 6: Gather ambient context (user profile, weather, logistics)
    ambient = _gather_ambient_context(events, store=store, web_search=web_search, user_tz=user_tz)

    # Step 7: Final synthesis via LLM
    console.print(f"\n  [bold]Synthesizing briefing via {model}...[/bold]")
    briefing_md = _synthesize_briefing(
        events_with_context, llm,
        ambient_context=ambient,
        user_context=user_context,
    )
    elapsed = time.time() - t0

    # Step 6: Store in derived.db
    event_ids = [e["id"] for e in events]
    derived.conn.execute(
        """INSERT INTO briefings (briefing_type, content, commitment_ids, prompt_version, model_used, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        ("meeting", briefing_md, json.dumps(event_ids), "briefing_v3_success_oriented", model, int(time.time())),
    )
    derived.conn.commit()

    console.print(f"\n  [bold green]Briefing ready[/bold green] ({elapsed:.1f}s, {len(events)} meetings)")

    return {
        "briefing": briefing_md,
        "events_count": len(events),
        "model": model,
        "elapsed_s": elapsed,
    }
