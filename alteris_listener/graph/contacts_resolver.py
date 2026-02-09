"""macOS Contacts reader — reads directly from AddressBook SQLite database.

Provides a phone/email → name lookup table for resolving
iMessage phone numbers and email-only contacts to display names.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_AB_DB_PATH = Path.home() / "Library" / "Application Support" / "AddressBook" / "AddressBook-v22.abcddb"


def _normalize_phone(phone: str) -> str:
    """Strip a phone number to digits only, with leading +1 for US."""
    digits = "".join(c for c in phone if c.isdigit())
    if len(digits) == 10:
        digits = "1" + digits
    if not digits.startswith("+"):
        digits = "+" + digits
    return digits


@dataclass
class ContactRecord:
    """A contact from macOS Contacts.app."""
    name: str
    emails: list[str] = field(default_factory=list)
    phones: list[str] = field(default_factory=list)


def _find_addressbook_dbs() -> list[Path]:
    """Find all AddressBook SQLite databases (root + per-source)."""
    dbs = []
    ab_dir = Path.home() / "Library" / "Application Support" / "AddressBook"

    # Root DB
    root = ab_dir / "AddressBook-v22.abcddb"
    if root.exists():
        dbs.append(root)

    # Per-source DBs (iCloud, Exchange, etc.)
    sources = ab_dir / "Sources"
    if sources.exists():
        for src_dir in sources.iterdir():
            candidate = src_dir / "AddressBook-v22.abcddb"
            if candidate.exists():
                dbs.append(candidate)

    return dbs


def read_contacts() -> list[ContactRecord]:
    """Read contacts from all AddressBook SQLite databases."""
    all_contacts: list[ContactRecord] = []

    dbs = _find_addressbook_dbs()
    if not dbs:
        logger.warning("No AddressBook databases found")
        return []

    for path in dbs:
        all_contacts.extend(_read_contacts_from_db(path))

    logger.info("Read %d contacts from %d AddressBook databases", len(all_contacts), len(dbs))
    return all_contacts


def _read_contacts_from_db(path: Path) -> list[ContactRecord]:
    """Read contacts from a single AddressBook SQLite database."""
    if not path.exists():
        logger.warning("AddressBook database not found at %s", path)
        return []

    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.OperationalError as exc:
        logger.warning("Cannot read AddressBook database: %s", exc)
        return []

    contacts_by_pk: dict[int, ContactRecord] = {}

    # Load all person records (Z_ENT=22 is ABPerson based on the schema)
    try:
        for row in conn.execute(
            """SELECT Z_PK, ZFIRSTNAME, ZLASTNAME, ZORGANIZATION, ZNICKNAME
               FROM ZABCDRECORD
               WHERE ZFIRSTNAME IS NOT NULL OR ZLASTNAME IS NOT NULL OR ZORGANIZATION IS NOT NULL"""
        ):
            first = row["ZFIRSTNAME"] or ""
            last = row["ZLASTNAME"] or ""
            org = row["ZORGANIZATION"] or ""
            nick = row["ZNICKNAME"] or ""

            name = f"{first} {last}".strip()
            if not name:
                name = org or nick or ""
            if not name:
                continue

            contacts_by_pk[row["Z_PK"]] = ContactRecord(name=name)
    except sqlite3.OperationalError as exc:
        logger.warning("Failed to read ZABCDRECORD: %s", exc)
        conn.close()
        return []

    # Load email addresses
    try:
        for row in conn.execute(
            "SELECT ZOWNER, ZADDRESSNORMALIZED, ZADDRESS FROM ZABCDEMAILADDRESS WHERE ZOWNER IS NOT NULL"
        ):
            pk = row["ZOWNER"]
            addr = row["ZADDRESSNORMALIZED"] or row["ZADDRESS"] or ""
            if pk in contacts_by_pk and addr:
                contacts_by_pk[pk].emails.append(addr.lower().strip())
    except sqlite3.OperationalError as exc:
        logger.warning("Failed to read ZABCDEMAILADDRESS: %s", exc)

    # Load phone numbers
    try:
        for row in conn.execute(
            "SELECT ZOWNER, ZFULLNUMBER FROM ZABCDPHONENUMBER WHERE ZOWNER IS NOT NULL"
        ):
            pk = row["ZOWNER"]
            phone = row["ZFULLNUMBER"] or ""
            if pk in contacts_by_pk and phone:
                contacts_by_pk[pk].phones.append(phone.strip())
    except sqlite3.OperationalError as exc:
        logger.warning("Failed to read ZABCDPHONENUMBER: %s", exc)

    conn.close()

    result = list(contacts_by_pk.values())
    logger.info("Read %d contacts from AddressBook database", len(result))
    return result


class ContactLookup:
    """Fast lookup table: phone/email → display name.

    Build once from Contacts.app database, then use to resolve
    iMessage phone numbers and email-only contacts to display names.
    """

    def __init__(self):
        self._by_email: dict[str, str] = {}
        self._by_phone: dict[str, str] = {}
        self._loaded = False

    def load(self) -> int:
        """Load contacts from AddressBook database. Returns count loaded."""
        contacts = read_contacts()

        for c in contacts:
            for email in c.emails:
                self._by_email[email.lower()] = c.name
            for phone in c.phones:
                normalized = _normalize_phone(phone)
                self._by_phone[normalized] = c.name

        self._loaded = True
        logger.info(
            "Contact lookup: %d email mappings, %d phone mappings",
            len(self._by_email), len(self._by_phone),
        )
        return len(contacts)

    def resolve_name(self, identifier: str) -> str | None:
        """Look up a display name by email or phone number.

        Tries email match first, then phone normalization.
        Returns None if not found.
        """
        if not self._loaded:
            self.load()

        # Try email
        lower = identifier.lower().strip()
        if lower in self._by_email:
            return self._by_email[lower]

        # Try phone
        normalized = _normalize_phone(identifier)
        if normalized in self._by_phone:
            return self._by_phone[normalized]

        return None

    @property
    def email_count(self) -> int:
        return len(self._by_email)

    @property
    def phone_count(self) -> int:
        return len(self._by_phone)
