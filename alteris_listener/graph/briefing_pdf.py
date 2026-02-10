"""Render a briefing markdown string to a styled PDF.

Usage:
    from alteris_listener.graph.briefing_pdf import render_briefing_pdf
    render_briefing_pdf(markdown_text, output_path, title="Weekly Briefing")

Or from CLI:
    alteris-listener graph brief --web --save ~/Desktop/brief.pdf
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    HRFlowable,
    KeepTogether,
    Table,
    TableStyle,
)


# ══════════════════════════════════════════════════════════════════
# Color palette
# ══════════════════════════════════════════════════════════════════

NAVY = colors.HexColor("#1a2332")
SLATE = colors.HexColor("#475569")
ACCENT = colors.HexColor("#2563eb")  # Blue accent
ACCENT_LIGHT = colors.HexColor("#eff6ff")  # Light blue background
WARN_BG = colors.HexColor("#fef3c7")  # Amber warning background
WARN_TEXT = colors.HexColor("#92400e")  # Amber warning text
BORDER = colors.HexColor("#e2e8f0")
MUTED = colors.HexColor("#94a3b8")
WHITE = colors.white
BLACK = colors.black


# ══════════════════════════════════════════════════════════════════
# Styles
# ══════════════════════════════════════════════════════════════════

def _build_styles():
    """Create the full stylesheet for briefing PDFs."""
    base = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "BriefTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            textColor=NAVY,
            spaceAfter=4,
            alignment=TA_LEFT,
        ),
        "subtitle": ParagraphStyle(
            "BriefSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            textColor=MUTED,
            spaceAfter=16,
        ),
        "event_title": ParagraphStyle(
            "EventTitle",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            textColor=NAVY,
            spaceBefore=0,
            spaceAfter=2,
        ),
        "event_meta": ParagraphStyle(
            "EventMeta",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            textColor=SLATE,
            spaceAfter=6,
        ),
        "bluf": ParagraphStyle(
            "BLUF",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            textColor=NAVY,
            spaceBefore=2,
            spaceAfter=6,
            leftIndent=0,
        ),
        "section_header": ParagraphStyle(
            "SectionHeader",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9.5,
            textColor=ACCENT,
            spaceBefore=8,
            spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "BriefBody",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            textColor=SLATE,
            spaceBefore=1,
            spaceAfter=3,
            leading=13,
        ),
        "bullet": ParagraphStyle(
            "BriefBullet",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            textColor=SLATE,
            spaceBefore=1,
            spaceAfter=2,
            leftIndent=16,
            bulletIndent=6,
            leading=13,
        ),
        "warning": ParagraphStyle(
            "Warning",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9.5,
            textColor=WARN_TEXT,
            spaceBefore=2,
            spaceAfter=2,
            leading=13,
        ),
        "blockquote": ParagraphStyle(
            "Blockquote",
            parent=base["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=9,
            textColor=SLATE,
            leftIndent=20,
            rightIndent=12,
            spaceBefore=4,
            spaceAfter=4,
            leading=13,
            borderPadding=(4, 8, 4, 8),
        ),
        "cross_event_title": ParagraphStyle(
            "CrossEventTitle",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            textColor=ACCENT,
            spaceBefore=12,
            spaceAfter=4,
        ),
        "holiday_line": ParagraphStyle(
            "HolidayLine",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            textColor=NAVY,
            spaceBefore=4,
            spaceAfter=4,
        ),
        "footer": ParagraphStyle(
            "Footer",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
    }
    return styles


# ══════════════════════════════════════════════════════════════════
# Markdown → Flowables parser
# ══════════════════════════════════════════════════════════════════

def _escape_xml(text: str) -> str:
    """Escape text for ReportLab XML paragraphs."""
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    return text


def _md_inline_to_rl(text: str) -> str:
    """Convert inline markdown formatting to ReportLab XML.

    Handles **bold**, *italic*, `code`, and ⚠️ emoji.
    """
    # Escape XML first
    text = _escape_xml(text)

    # Bold: **text** → <b>text</b>
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # Italic: *text* → <i>text</i> (but not inside bold)
    text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", text)
    # Code: `text` → <font name="Courier" size="8.5">text</font>
    text = re.sub(r"`([^`]+?)`", r'<font name="Courier" size="8.5">\1</font>', text)

    return text


def _parse_briefing_to_flowables(md: str, styles: dict) -> list:
    """Parse the briefing markdown into ReportLab flowable objects."""
    flowables = []
    lines = md.split("\n")
    i = 0
    current_event_items = []  # collect items for KeepTogether

    def _flush_event():
        nonlocal current_event_items
        if current_event_items:
            flowables.append(KeepTogether(current_event_items[:6]))
            if len(current_event_items) > 6:
                flowables.extend(current_event_items[6:])
            current_event_items = []

    while i < len(lines):
        line = lines[i].rstrip()

        # Skip empty lines
        if not line:
            i += 1
            continue

        # ── Horizontal rule (event separator) ──
        if re.match(r"^-{3,}$", line):
            _flush_event()
            flowables.append(Spacer(1, 6))
            flowables.append(HRFlowable(
                width="100%", thickness=0.5, color=BORDER,
                spaceBefore=4, spaceAfter=8,
            ))
            i += 1
            continue

        # ── H2: Event title or Cross-Event Notes ──
        if line.startswith("## "):
            _flush_event()
            title_text = line[3:].strip()

            # Cross-event notes section
            if "cross-event" in title_text.lower():
                current_event_items.append(Spacer(1, 4))
                current_event_items.append(Paragraph(
                    _md_inline_to_rl(title_text),
                    styles["cross_event_title"],
                ))
            else:
                current_event_items.append(Paragraph(
                    _md_inline_to_rl(title_text),
                    styles["event_title"],
                ))
            i += 1
            continue

        # ── H3: Section header within an event ──
        if line.startswith("### "):
            header_text = line[4:].strip()
            current_event_items.append(Paragraph(
                _md_inline_to_rl(header_text),
                styles["section_header"],
            ))
            i += 1
            continue

        # ── Holiday single-line (📌 prefix) ──
        if line.startswith("📌"):
            _flush_event()
            text = _md_inline_to_rl(line)
            flowables.append(Paragraph(text, styles["holiday_line"]))
            i += 1
            continue

        # ── Metadata line: **When:** / **With:** / **Where:** ──
        if re.match(r"^\*\*(?:When|With|Where|Duration|Link|RSVP).*?:", line):
            text = _md_inline_to_rl(line)
            current_event_items.append(Paragraph(text, styles["event_meta"]))
            i += 1
            continue

        # ── BLUF line ──
        if line.startswith("**BLUF:**"):
            bluf_text = line.replace("**BLUF:**", "").strip()
            current_event_items.append(Paragraph(
                f"<b>BLUF:</b> {_md_inline_to_rl(bluf_text)}",
                styles["bluf"],
            ))
            i += 1
            continue

        # ── Section headers: **Your agenda:** etc. ──
        section_match = re.match(
            r"^\*\*(Their agenda|Your agenda|What changed|Open items|People|"
            r"Gift ideas|Logistics|Prep|Cross-Event Notes|Action|Draft message).*?:\*\*\s*(.*)",
            line,
        )
        if section_match:
            header = section_match.group(1)
            rest = section_match.group(2).strip()
            current_event_items.append(Paragraph(
                _md_inline_to_rl(f"**{header}**"),
                styles["section_header"],
            ))
            if rest:
                current_event_items.append(Paragraph(
                    _md_inline_to_rl(rest),
                    styles["body"],
                ))
            i += 1
            continue

        # ── Warning lines (⚠️) ──
        if "⚠️" in line or "⚠" in line:
            text = _md_inline_to_rl(line.lstrip("- •"))
            # Wrap in a colored table for visual emphasis
            warning_para = Paragraph(text, styles["warning"])
            warning_table = Table(
                [[warning_para]],
                colWidths=[6.5 * inch],
            )
            warning_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), WARN_BG),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("ROUNDEDCORNERS", [4, 4, 4, 4]),
            ]))
            current_event_items.append(warning_table)
            i += 1
            continue

        # ── Blockquote (▌ prefix — used for draft messages) ──
        if line.startswith("▌") or line.startswith(">"):
            quote_text = line.lstrip("▌> ").strip()
            # Collect multi-line blockquotes
            while i + 1 < len(lines) and (lines[i + 1].startswith("▌") or lines[i + 1].startswith(">")):
                i += 1
                quote_text += " " + lines[i].lstrip("▌> ").strip()

            quote_para = Paragraph(
                f"<i>{_md_inline_to_rl(quote_text)}</i>",
                styles["blockquote"],
            )
            quote_table = Table(
                [[quote_para]],
                colWidths=[6.2 * inch],
            )
            quote_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), ACCENT_LIGHT),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("LINEBEFORESTARTX", (0, 0), (0, -1), 3, ACCENT),
            ]))
            current_event_items.append(Spacer(1, 2))
            current_event_items.append(quote_table)
            current_event_items.append(Spacer(1, 2))
            i += 1
            continue

        # ── Numbered list items ──
        numbered_match = re.match(r"^\s*(\d+)[\.\)]\s+(.*)", line)
        if numbered_match:
            num = numbered_match.group(1)
            text = numbered_match.group(2).strip()
            current_event_items.append(Paragraph(
                f"<b>{num}.</b> {_md_inline_to_rl(text)}",
                styles["bullet"],
            ))
            i += 1
            continue

        # ── Bullet points ──
        bullet_match = re.match(r"^\s*[-•*]\s+(.*)", line)
        if bullet_match:
            text = bullet_match.group(1).strip()
            current_event_items.append(Paragraph(
                f"&bull; {_md_inline_to_rl(text)}",
                styles["bullet"],
            ))
            i += 1
            continue

        # ── Default: body paragraph ──
        text = _md_inline_to_rl(line)
        if text.strip():
            current_event_items.append(Paragraph(text, styles["body"]))
        i += 1

    _flush_event()
    return flowables


# ══════════════════════════════════════════════════════════════════
# Header / footer
# ══════════════════════════════════════════════════════════════════

def _add_header_footer(canvas_obj, doc, title: str, generated_at: str):
    """Draw header and footer on each page."""
    canvas_obj.saveState()
    width, height = letter

    # Header line
    canvas_obj.setStrokeColor(ACCENT)
    canvas_obj.setLineWidth(2)
    canvas_obj.line(
        doc.leftMargin,
        height - doc.topMargin + 16,
        width - doc.rightMargin,
        height - doc.topMargin + 16,
    )

    # Footer
    canvas_obj.setFont("Helvetica", 7)
    canvas_obj.setFillColor(MUTED)
    canvas_obj.drawCentredString(
        width / 2,
        doc.bottomMargin - 16,
        f"Alteris Briefing  ·  {generated_at}  ·  Page {doc.page}",
    )

    canvas_obj.restoreState()


# ══════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════

def render_briefing_pdf(
    markdown: str,
    output_path: str | Path,
    title: str = "Meeting Briefing",
    generated_at: str | None = None,
) -> Path:
    """Render a briefing markdown string to a styled PDF.

    Args:
        markdown: The briefing text (as generated by _synthesize_briefing)
        output_path: Where to write the PDF
        title: Title shown at top of PDF
        generated_at: Timestamp string. Defaults to now.

    Returns:
        Path to the generated PDF.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if generated_at is None:
        generated_at = datetime.now().strftime("%B %d, %Y at %I:%M %p")

    styles = _build_styles()

    # Build the document
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        title=title,
        author="Alteris",
    )

    # Build flowables
    story = []

    # Title block
    story.append(Paragraph(title, styles["title"]))
    story.append(Paragraph(
        f"Generated {generated_at}",
        styles["subtitle"],
    ))
    story.append(HRFlowable(
        width="100%", thickness=1, color=ACCENT,
        spaceBefore=0, spaceAfter=12,
    ))

    # Parse the briefing content
    content = _parse_briefing_to_flowables(markdown, styles)
    story.extend(content)

    # Build with header/footer
    doc.build(
        story,
        onFirstPage=lambda c, d: _add_header_footer(c, d, title, generated_at),
        onLaterPages=lambda c, d: _add_header_footer(c, d, title, generated_at),
    )

    return output_path
