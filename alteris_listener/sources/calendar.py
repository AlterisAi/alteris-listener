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
    if status not in (3, 4):
        logger.warning("Calendar access not granted (status=%s)", status)
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

        body_parts = [title]
        if location:
            body_parts.append(f"Location: {location}")
        if is_all_day:
            body_parts.append("All day event")
        else:
            body_parts.append(f"{start_dt.strftime('%H:%M')} - {end_dt.strftime('%H:%M')}")
        if notes:
            body_parts.append(notes[:500])

        messages.append(Message(
            source="calendar",
            sender=cal_name,
            recipient="me",
            subject=title,
            body="\n".join(body_parts),
            timestamp=start_dt,
            metadata={
                "calendar": cal_name,
                "is_all_day": is_all_day,
                "end": end_dt.isoformat(),
                "location": location,
            },
        ))

    return messages
