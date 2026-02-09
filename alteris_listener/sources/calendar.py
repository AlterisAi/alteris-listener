"""Calendar reader — reads from macOS Calendar via EventKit."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List

from alteris_listener.sources.base import Message

logger = logging.getLogger(__name__)


def read_upcoming_events(days_ahead: int = 7, days_behind: int = 1) -> List[Message]:
    """Read calendar events from macOS Calendar via EventKit.

    Returns events as Message objects for uniform processing.
    """
    try:
        import EventKit
        from Foundation import NSDate, NSCalendar, NSDateComponents
    except ImportError:
        logger.warning("EventKit not available — install pyobjc-framework-EventKit")
        return []

    store = EventKit.EKEventStore.alloc().init()

    status = EventKit.EKEventStore.authorizationStatusForEntityType_(EventKit.EKEntityTypeEvent)

    # Status 0 = not determined — request access to trigger the macOS permission prompt
    if status == 0:
        import threading
        granted_event = threading.Event()
        grant_result = [False]

        def callback(granted, error):
            grant_result[0] = granted
            granted_event.set()

        store.requestFullAccessToEventsWithCompletion_(callback)
        granted_event.wait(timeout=30)

        if not grant_result[0]:
            logger.warning(
                "Calendar access denied. Grant access in: "
                "System Settings → Privacy & Security → Calendars"
            )
            return []
    elif status not in (3, 4):
        logger.warning(
            "Calendar access not granted (status=%s). Grant access in: "
            "System Settings → Privacy & Security → Calendars",
            status,
        )
        return []

    cal = NSCalendar.currentCalendar()

    start_comp = NSDateComponents.alloc().init()
    start_comp.setDay_(-days_behind)
    start_date = cal.dateByAddingComponents_toDate_options_(start_comp, NSDate.date(), 0)

    end_comp = NSDateComponents.alloc().init()
    end_comp.setDay_(days_ahead)
    end_date = cal.dateByAddingComponents_toDate_options_(end_comp, NSDate.date(), 0)

    predicate = store.predicateForEventsWithStartDate_endDate_calendars_(start_date, end_date, None)
    events = store.eventsMatchingPredicate_(predicate)

    if not events:
        return []

    messages = []
    for event in events:
        start_ts = event.startDate().timeIntervalSince1970()
        end_ts = event.endDate().timeIntervalSince1970()
        start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)

        title = str(event.title() or "")
        notes = str(event.notes() or "") if event.notes() else ""
        location = str(event.location() or "") if event.location() else ""
        cal_name = str(event.calendar().title()) if event.calendar() else ""
        is_all_day = bool(event.isAllDay())

        # Extract organizer
        organizer_name = ""
        organizer_email = ""
        if event.organizer():
            org = event.organizer()
            organizer_name = str(org.name() or "")
            try:
                organizer_email = str(org.emailAddress() or "")
            except AttributeError:
                pass

        # Extract attendees with RSVP status
        attendee_list = []
        ek_attendees = event.attendees()
        if ek_attendees and len(ek_attendees) > 0:
            status_map = {
                0: "pending", 1: "accepted", 2: "declined",
                3: "tentative", 4: "delegated", 5: "completed",
                6: "in-process",
            }
            role_map = {
                0: "unknown", 1: "required", 2: "optional",
                3: "chair", 4: "non-participant",
            }
            for att in ek_attendees:
                att_name = str(att.name() or "")
                att_status = status_map.get(att.participantStatus(), "unknown")
                att_role = role_map.get(att.participantRole(), "unknown")
                # Extract email from URL (mailto:...)
                att_email = ""
                att_url = att.URL()
                if att_url:
                    url_str = str(att_url)
                    if "mailto:" in url_str:
                        att_email = url_str.split("mailto:")[-1]
                attendee_list.append({
                    "name": att_name,
                    "email": att_email,
                    "status": att_status,
                    "role": att_role,
                })

        # Extract URL
        event_url = ""
        if event.URL():
            event_url = str(event.URL())

        # Recurrence info
        is_recurring = bool(event.hasRecurrenceRules())
        recurrence_desc = ""
        if is_recurring and event.recurrenceRules():
            freq_map = {0: "daily", 1: "weekly", 2: "monthly", 3: "yearly"}
            for rule in event.recurrenceRules():
                freq = freq_map.get(rule.frequency(), "unknown")
                interval = rule.interval()
                recurrence_desc = f"{freq}" if interval == 1 else f"every {interval} {freq}"

        # Event status
        ev_status_map = {0: "none", 1: "confirmed", 2: "tentative", 3: "cancelled"}
        event_status = ev_status_map.get(event.status(), "unknown")

        # Build rich body text
        body_parts = [title]
        if location:
            body_parts.append(f"Location: {location}")
        if is_all_day:
            body_parts.append("All day event")
        else:
            body_parts.append(f"{start_dt.strftime('%H:%M')} - {end_dt.strftime('%H:%M')}")
        if organizer_name:
            body_parts.append(f"Organizer: {organizer_name}" +
                              (f" ({organizer_email})" if organizer_email else ""))
        if attendee_list:
            att_strs = []
            for a in attendee_list:
                s = a["name"] or a["email"] or "unknown"
                if a["status"] not in ("pending", "unknown"):
                    s += f" [{a['status']}]"
                att_strs.append(s)
            body_parts.append(f"Attendees: {', '.join(att_strs)}")
        if recurrence_desc:
            body_parts.append(f"Recurrence: {recurrence_desc}")
        if event_url:
            body_parts.append(f"URL: {event_url}")
        if notes:
            body_parts.append(notes[:1000])

        messages.append(Message(
            source="calendar",
            sender=organizer_email or cal_name,
            recipient="me",
            subject=title,
            body="\n".join(body_parts),
            timestamp=start_dt,
            metadata={
                "calendar": cal_name,
                "is_all_day": is_all_day,
                "end": end_dt.isoformat(),
                "location": location,
                "organizer_name": organizer_name,
                "organizer_email": organizer_email,
                "attendees": attendee_list,
                "event_url": event_url,
                "is_recurring": is_recurring,
                "recurrence": recurrence_desc,
                "status": event_status,
            },
        ))

    # Deduplicate by (title, start_time) — same event synced across calendars
    seen: set[tuple[str, float]] = set()
    deduped: list[Message] = []
    for msg in messages:
        key = (msg.subject.strip().lower(), msg.timestamp.timestamp())
        if key not in seen:
            seen.add(key)
            deduped.append(msg)

    # Collapse recurring events: keep only the next upcoming and most recent past
    # occurrence for each recurring event title
    import time as _time
    now_ts = _time.time()
    recurring_groups: dict[str, list[Message]] = {}
    non_recurring: list[Message] = []

    for msg in deduped:
        meta = msg.metadata or {}
        if meta.get("is_recurring"):
            title_key = msg.subject.strip().lower()
            recurring_groups.setdefault(title_key, []).append(msg)
        else:
            non_recurring.append(msg)

    collapsed: list[Message] = list(non_recurring)
    for title_key, occurrences in recurring_groups.items():
        past = [m for m in occurrences if m.timestamp.timestamp() <= now_ts]
        future = [m for m in occurrences if m.timestamp.timestamp() > now_ts]
        # Keep most recent past occurrence
        if past:
            past.sort(key=lambda m: m.timestamp.timestamp(), reverse=True)
            collapsed.append(past[0])
        # Keep next upcoming occurrence
        if future:
            future.sort(key=lambda m: m.timestamp.timestamp())
            collapsed.append(future[0])

    logger.info(
        "Calendar: %d raw, %d after cross-cal dedup, %d after recurring collapse",
        len(messages), len(deduped), len(collapsed),
    )

    # Extract shared calendar signal — calendars named "X and Y" where
    # one name matches the user indicate family/partner relationship.
    # Store as metadata on the messages for downstream tier computation.
    shared_calendars: dict[str, str] = {}
    user_first_names = {"aniruddha", "ani"}  # TODO: make configurable
    for msg in collapsed:
        cal = (msg.metadata or {}).get("calendar", "")
        if " and " in cal.lower():
            parts = [p.strip().lower() for p in cal.lower().split(" and ")]
            for part in parts:
                if part in user_first_names:
                    other = [p for p in parts if p != part]
                    if other:
                        shared_calendars[cal] = other[0].title()
    if shared_calendars:
        for msg in collapsed:
            cal = (msg.metadata or {}).get("calendar", "")
            if cal in shared_calendars:
                msg.metadata["shared_calendar_with"] = shared_calendars[cal]
                msg.metadata["family_signal"] = True
        logger.info(
            "Shared calendars detected: %s",
            ", ".join(f"{k} → {v}" for k, v in shared_calendars.items()),
        )

    return collapsed
