"""
email_cleaner.py

Utilities to clean email message bodies before task extraction.

Goals:
- Keep: main message content, signatures, phone numbers, useful URLs.
- Remove: quoted history from earlier messages, generic legal disclaimers,
  long tracking wrappers around URLs, and obvious boilerplate.

This is deliberately conservative: when in doubt, we keep content.
"""

import re
from typing import List, Tuple


# ------------------------------
# REGEXES
# ------------------------------

AUTO_LINK_RE = re.compile(r"<(https?://[^>]+)>")
URL_RE = re.compile(r"https?://[^\s>]+")
PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?\d[\d\-\(\) ]{6,}\d)(?!\w)"
)
DATE_RE = re.compile(
    r"\b(?:\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b|"   # numeric 11/20
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}\b",
    re.IGNORECASE
)
TASKY_LINE_RE = re.compile(r"^\s*(\d+\.|\-)\s+.+")

SIGNATURE_STARTERS = [
    "best",
    "thanks",
    "thank you",
    "regards",
    "kind regards",
    "warmly",
    "sincerely",
    "cheers",
    "with appreciation",
]


# ------------------------------
# UTILITIES
# ------------------------------

def unwrap_auto_links(line: str) -> str:
    """
    Converts Gmail auto-linked URLs from angle bracket format to normal format.
    
    Also handles cases where URLs split phone numbers:
    Example: '(206) <https://...> 282-2373' becomes '(206) 282-2373 https://...'
    """
    # Replace <url> with url
    line = AUTO_LINK_RE.sub(lambda m: f" {m.group(1)} ", line)
    
    # Detect if URL splits a number: "(206)  URL  282-2373"
    tokens = line.split()
    fixed = []
    i = 0
    
    while i < len(tokens):
        tok = tokens[i]
        if URL_RE.match(tok):
            # Check neighbors
            if (i - 1 >= 0 and re.search(r"\(\d{3}\)$", tokens[i-1])) and \
               (i + 1 < len(tokens) and re.match(r"^\d{3,4}\-?\d+$", tokens[i+1])):
                # Merge example:
                # ['(206)', 'URL', '282-2373'] -> '(206) 282-2373'
                merged_phone = tokens[i-1] + " " + tokens[i+1]
                fixed[-1] = merged_phone  # replace previous
                fixed.append(tok)  # keep URL after phone
                i += 2
                continue
        
        fixed.append(tok)
        i += 1
    
    return " ".join(fixed)


def collapse_blank_lines(lines: List[str]) -> List[str]:
    """Remove duplicate blank lines, keeping at most one blank line."""
    out = []
    prev_blank = False
    
    for l in lines:
        if not l.strip():
            if prev_blank:
                continue
            prev_blank = True
            out.append("")
        else:
            prev_blank = False
            out.append(l)
    
    return out


def split_signature(lines: List[str]) -> Tuple[List[str], List[str]]:
    """
    Split message into (body, signature_block).
    
    Constraints:
      - Only search bottom 35% of message
      - Up to the last 12 lines
    """
    if len(lines) <= 6:
        return lines, []
    
    start_scan = int(len(lines) * 0.65)
    candidate = lines[start_scan:]
    sig_idx = None
    
    # Scan backwards bottom 12 lines only
    scan_limit = candidate[-12:]
    
    for i in range(len(scan_limit)-1, -1, -1):
        raw = scan_limit[i].strip().lower()
        for starter in SIGNATURE_STARTERS:
            if raw.startswith(starter):
                sig_idx = len(lines) - len(scan_limit) + i
                break
        if sig_idx is not None:
            break
    
    if sig_idx is None:
        return lines, []
    
    return lines[:sig_idx], lines[sig_idx:]


def signature_has_relevant_info(sig_lines: List[str]) -> bool:
    """
    Keep signature if it contains any metadata relevant to tasks.
    """
    for raw in sig_lines:
        line = unwrap_auto_links(raw)
        if PHONE_RE.search(line):
            return True
        if URL_RE.search(line):
            return True
        if DATE_RE.search(line):
            return True
        if TASKY_LINE_RE.match(line):
            return True
    return False


def clean_signature(sig_lines: List[str]) -> List[str]:
    """
    Reduce signature to only relevant information:
    - phone numbers
    - URLs
    - dates
    - action-like lines
    """
    cleaned = []
    for raw in sig_lines:
        line = unwrap_auto_links(raw)
        if PHONE_RE.search(line) or URL_RE.search(line) or DATE_RE.search(line):
            cleaned.append(line.strip())
    return cleaned


def clean_email_body(body: str) -> str:
    """
    Remove quoted/replied text from email body, keeping only the new message content.
    
    Intelligently removes signatures but preserves phone numbers, URLs, dates,
    and task-like content if present in signatures.
    
    This removes conversation history that appears in email replies, such as:
    - Lines starting with "On ... wrote:"
    - Quoted lines starting with ">"
    - Forwarded email headers (From:, Sent:, To:, Subject:)
    """
    if not body:
        return ""
    
    # First, decode HTML entities that might interfere with pattern matching
    body = body.replace("&lt;", "<").replace("&gt;", ">").replace("&nbsp;", " ").replace("&amp;", "&")
    
    body = body.replace("\r\n", "\n").strip()
    raw_lines = body.split("\n")
    
    # Pass 1: unwrap auto links everywhere
    lines = [unwrap_auto_links(l) for l in raw_lines]
    
    # Pass 2: remove quoted sections and Outlook blocks
    cleaned = []
    skip_outlook = False
    outlook_header = re.compile(r"^(From|Sent|To|Subject):", re.IGNORECASE)
    
    # Patterns for separator lines and disclaimers
    separator_line = re.compile(r"^[-_=]{3,}\s*$")  # Lines with 3+ dashes, underscores, or equals
    external_disclaimer = re.compile(r".*[Oo]riginated.*[Oo]utside.*", re.IGNORECASE)  # "This message originated outside..."
    
    # More flexible "On ... wrote:" pattern - handles various date formats
    # Matches patterns like:
    # - "On Sep 14, 2025, at 13:44, Name <email> wrote:"
    # - "On Sun, 9 Nov 2025 at 15:05, Name <email> wrote:"
    # - "usOn Sep 14..." (when HTML conversion loses spaces - "On" followed by space + month)
    # Pattern: "On" (anywhere, even mid-word like "usOn") + space + date info + "wrote"
    # We match "On" followed by space and then date patterns to avoid false positives
    on_wrote_pattern = re.compile(
        r"On\s+(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)?,?\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}.*?wrote:?",
        re.IGNORECASE
    )
    # Simpler pattern for "On ... wrote:" - catches "On" followed by date-like patterns and "wrote"
    # This handles "On Sun, 9 Nov 2025 at 15:05" where day comes before month
    on_wrote_simple = re.compile(
        r"On\s+.+?(?:\d{4}|\d{1,2}:\d{2}).*?wrote:?",
        re.IGNORECASE
    )
    # Even simpler: just "On" + month name + date + year + "wrote" (catches edge cases like "usOn Sep 14, 2025")
    on_wrote_fallback = re.compile(
        r"On\s+[A-Z][a-z]+\s+\d{1,2},?\s+\d{4}.*?wrote:?",
        re.IGNORECASE
    )
    
    i = 0
    consecutive_blank_lines = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Track consecutive blank lines - if we see blank lines followed by "On ... wrote:", break
        if not stripped:
            consecutive_blank_lines += 1
            # If we have blank lines, look ahead to see if next non-empty line is "On ... wrote:"
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                next_stripped = lines[j].strip()
                # Check if next line matches "On ... wrote:" pattern
                if (on_wrote_pattern.match(next_stripped) or 
                    on_wrote_simple.match(next_stripped) or 
                    on_wrote_fallback.match(next_stripped)):
                    # Blank lines followed by "On ... wrote:" - this is start of quoted section
                    break
            # If not a quote header, keep the blank line (but limit consecutive blanks)
            if consecutive_blank_lines <= 2:
                cleaned.append(line)
            i += 1
            continue
        else:
            consecutive_blank_lines = 0
        
        # Quoted reply (lines starting with >)
        if stripped.startswith(">"):
            i += 1
            continue
        
        # "On ... wrote:" patterns (flexible matching) - check both start of line and anywhere in line
        # Sometimes these patterns appear mid-line when HTML is poorly converted
        if on_wrote_pattern.match(stripped) or on_wrote_simple.match(stripped) or on_wrote_fallback.match(stripped):
            break
        
        # Also check if pattern appears anywhere in the line (for poorly formatted HTML)
        # This handles cases like "...usOn Sep 14, 2025, at 13:44..."
        match = None
        for pattern in [on_wrote_pattern, on_wrote_simple, on_wrote_fallback]:
            match = pattern.search(line)
            if match:
                break
        
        if match:
            # Keep only the part before the match
            before_match = line[:match.start()].rstrip()
            if before_match:
                cleaned.append(before_match)
            break
        
        # Check for separator + disclaimer pattern (common before quoted content)
        if separator_line.match(stripped):
            # Look ahead to see if next non-empty line is a disclaimer or "On ... wrote:"
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                next_stripped = lines[j].strip()
                if (external_disclaimer.search(next_stripped) or 
                    on_wrote_pattern.match(next_stripped) or 
                    on_wrote_simple.match(next_stripped) or
                    on_wrote_fallback.match(next_stripped)):
                    # Found separator followed by disclaimer or reply header - this is start of quoted section
                    break
        
        # External message disclaimers - these often appear before quoted content
        if external_disclaimer.search(stripped):
            # Look ahead to see if next non-empty line is another separator or "On ... wrote:"
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                next_stripped = lines[j].strip()
                if (separator_line.match(next_stripped) or 
                    on_wrote_pattern.match(next_stripped) or 
                    on_wrote_simple.match(next_stripped) or
                    on_wrote_fallback.match(next_stripped)):
                    # Disclaimer followed by separator or reply header - break here
                    break
        
        # Outlook header blocks
        if outlook_header.match(stripped):
            skip_outlook = True
            i += 1
            continue
        if skip_outlook:
            if stripped == "":
                skip_outlook = False
            i += 1
            continue
        
        cleaned.append(line)
        i += 1
    
    # Pass 3: detect & split signature
    body_lines, sig_lines = split_signature(cleaned)
    
    # If signature contains relevant metadata, keep a cleaned version
    if signature_has_relevant_info(sig_lines):
        sig_clean = clean_signature(sig_lines)
        combined = body_lines + [""] + sig_clean
    else:
        combined = body_lines
    
    # Final cleanup
    combined = collapse_blank_lines(combined)
    
    return "\n".join(l.rstrip() for l in combined).strip()
