"""Alteris Voice Agent — Gemini Live API with native audio.

Architecture (concurrent two-phase):
    Phase 1 (background, ~15s): Briefing pipeline via Flash 3
        - While greeting plays via Live API
        - Gathers events, context, triage
        - Generates a voice summary + open items for the voice model
    Phase 2 (voice, 3-5 min): Conversational open-item resolution
        - Walk through open items, resolve each via Q&A
        - Tools available for follow-ups
    On exit: Flash 3 synthesizes full briefing with answers → PDF

Usage:
    alteris-listener graph voice [--days 7]
    alteris-listener graph voice --vertex

Requirements:
    pip install google-genai pyaudio
"""

from __future__ import annotations

import asyncio
import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

# ══════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════

DEV_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
VERTEX_MODEL = "gemini-live-2.5-flash-native-audio"

SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024
DEFAULT_DAYS = 7


# ══════════════════════════════════════════════════════════════════
# Tool declarations (for interactive follow-ups during voice)
# ══════════════════════════════════════════════════════════════════

TOOL_DECLARATIONS = [
    {
        "name": "search_graph",
        "description": (
            "Search the user's knowledge graph for emails, messages, "
            "and notes related to a topic or keyword. Use when the user "
            "asks to dig deeper on something."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Topic, person name, project, or keyword",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results (default 10)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_emails_from_person",
        "description": "Get recent emails from a specific person.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Person's name"},
                "email": {"type": "string", "description": "Email (optional)"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "get_messages_with_person",
        "description": "Get recent iMessage/text conversations with a person.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Person's name"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "get_commitments",
        "description": "Get open commitments and action items related to a topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Topic or meeting name"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "save_decision",
        "description": (
            "Record a decision or answer the user gave during this session. "
            "Call after each substantive answer or decision."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "event": {"type": "string", "description": "Which event or topic"},
                "decision": {"type": "string", "description": "What was decided or answered"},
            },
            "required": ["event", "decision"],
        },
    },
]


# ══════════════════════════════════════════════════════════════════
# System prompts
# ══════════════════════════════════════════════════════════════════

GREETING_PROMPT = """\
You are Alteris, an AI chief of staff. You are starting a voice \
briefing session.

Say a warm good morning. Tell them you're pulling their calendar, \
emails, and messages for the next {days} days right now. Say you'll \
be back in about 15 seconds with some questions to make sure the \
briefing covers everything they need. Keep it to 3-4 natural \
sentences. Be warm and personable, like a great EA who genuinely \
cares about the person's day.
"""

# This prompt gets the voice summary + open items injected at runtime
CONVERSATION_PROMPT = """\
You are Alteris, an AI chief of staff having a voice conversation \
with the user about their upcoming week.

You have already analyzed their calendar, emails, and messages. \
Below is a summary of what you found and the open items that need \
the user's input.

YOUR JOB: Walk through the open items conversationally. For each one:
1. Give enough context so the user understands WHY you're asking — \
   mention specific people, dates, or details from the analysis.
2. Ask the clear question.
3. Listen to their answer. Acknowledge it warmly and substantively \
   (not just "got it" — show you understood what they said and why \
   it matters).
4. If their answer raises a follow-up (e.g., they need to send a \
   message, buy something, confirm something), offer to help.
5. Use save_decision to record each answer.
6. Transition naturally to the next item.

START by giving a brief overview of the week — what's coming \
up, what the big themes are, which items need their attention. \
Do NOT say good morning or greet the user — you already did that. \
Jump straight into the overview, then move into the open items \
one by one.

WHEN ALL ITEMS ARE RESOLVED, ask if there's anything else they \
want to dig into. If they mention a topic, use search_graph or \
other tools to find relevant info. When they're done, say you'll \
compile the full briefing and save it to their desktop.

STYLE:
- Warm, conversational, substantive. Not terse, not a readout.
- Give context with each question — don't make them guess why \
  you're asking.
- 3-5 sentences per item is fine. Natural spoken language.
- No bullet points, markdown, numbers, or formatting.
- Use names, dates, and specifics. Show you've done the homework.
- If an open item involves a message to draft, offer to help \
  compose it right there.

IMPORTANT RULES:
- NEVER mention that attendees "declined" a meeting. Calendar RSVP \
  data is often stale or wrong. Treat all meetings as happening.
- Do not read out raw email subjects, IDs, or dates. Synthesize.
- Focus on what the user needs to DO, not what happened.
- If the user interrupts you, pause and listen. Then pick up where \
  you left off or address their question before continuing.
- Keep each spoken segment to 3-5 sentences, then pause. This \
  gives the user a chance to respond naturally.

VOICE SUMMARY:
{voice_summary}

OPEN ITEMS TO RESOLVE:
{open_items}
"""


# ══════════════════════════════════════════════════════════════════
# Voice summary generator (runs as part of stage A)
# ══════════════════════════════════════════════════════════════════

VOICE_SUMMARY_SYSTEM = """\
You are preparing a voice briefing for an AI chief of staff to \
deliver conversationally to a busy professional. Given the event \
context below, produce TWO things:

1. VOICE_SUMMARY: A 4-6 sentence natural-language overview of the \
   user's upcoming week. Mention the key events, any scheduling \
   conflicts or logistics issues, and the overall shape of the week. \
   Write as if speaking aloud — warm, specific, no formatting.

   IMPORTANT: Ignore RSVP/decline status of attendees. Calendar \
   declines are often stale or wrong. Do NOT mention that people \
   declined. Focus on the substance of each event.

2. OPEN_ITEMS: A list of open items that need the user's input. \
   Each item should have:
   - event: which event this relates to
   - context: 2-3 sentences of background (specific names, dates, \
     what happened) so the voice agent can explain WHY it's asking
   - question: the specific question to ask
   - suggested_action: what to do with the answer (e.g., "draft \
     message", "note for briefing", "confirm logistics")

   Do NOT create open items about RSVP status or whether a meeting \
   is still happening based on declines. Focus on items where the \
   user's input would actually change the briefing or resolve an \
   open loop. Skip trivial things. Aim for 2-5 open items max.

Respond in JSON:
{
  "voice_summary": "...",
  "open_items": [
    {
      "event": "...",
      "context": "...",
      "question": "...",
      "suggested_action": "..."
    }
  ]
}
"""


# ══════════════════════════════════════════════════════════════════
# Tool handler
# ══════════════════════════════════════════════════════════════════

class ToolHandler:

    def __init__(self):
        self.decisions: list[dict] = []

    def _graph_query(self, query_type: str, **params) -> list[dict]:
        from alteris_listener.graph.briefing import (
            _execute_graph_query, _load_user_profile,
        )
        from alteris_listener.graph.store import GraphStore
        from alteris_listener.graph.derived_store import DerivedStore
        store = GraphStore()
        derived = DerivedStore()
        profile = _load_user_profile()
        user_tz = profile.get("timezone", "America/Los_Angeles")
        try:
            return _execute_graph_query(
                {"query_type": query_type, "params": params},
                store, derived, user_tz,
            )
        finally:
            store.close()
            derived.close()

    async def handle(self, name: str, args: dict[str, Any]) -> dict:
        handlers = {
            "search_graph": self._search_graph,
            "get_emails_from_person": self._get_emails_from_person,
            "get_messages_with_person": self._get_messages_with_person,
            "get_commitments": self._get_commitments,
            "save_decision": self._save_decision,
        }
        handler = handlers.get(name)
        if not handler:
            return {"error": f"Unknown function: {name}"}
        try:
            return await asyncio.to_thread(handler, **args)
        except Exception as e:
            traceback.print_exc()
            return {"error": str(e)}

    def _search_graph(self, query: str, max_results: int = 10) -> dict:
        keywords = query.split()[:5]
        results = self._graph_query("topic_search", keywords=keywords)
        return {"results": results[:int(max_results)], "count": len(results)}

    def _get_emails_from_person(self, name: str, email: str = None) -> dict:
        results = self._graph_query("person_emails", name=name, email=email or "")
        return {"emails": results, "count": len(results), "person": name}

    def _get_messages_with_person(self, name: str) -> dict:
        results = self._graph_query("person_messages", name=name)
        return {"messages": results, "count": len(results), "person": name}

    def _get_commitments(self, query: str) -> dict:
        keywords = query.split()[:5]
        results = self._graph_query("commitments_search", keywords=keywords)
        return {"commitments": results, "count": len(results)}

    def _save_decision(self, event: str, decision: str) -> dict:
        entry = {"event": event, "decision": decision,
                 "timestamp": datetime.now().isoformat()}
        self.decisions.append(entry)
        print(f"  💾 Decision: [{event}] {decision}")
        return {"saved": True, "total_decisions": len(self.decisions)}


# ══════════════════════════════════════════════════════════════════
# Pipeline stages
# ══════════════════════════════════════════════════════════════════

def _run_pipeline_stage_a(days: int) -> dict:
    """Events + context + triage + voice summary + open items.

    Returns dict with everything needed for both voice and synthesis.
    """
    import re
    import time
    from alteris_listener.graph.store import GraphStore
    from alteris_listener.graph.derived_store import DerivedStore
    from alteris_listener.graph.briefing import (
        _get_upcoming_events, _load_user_profile,
        _gather_meeting_context, _agentic_context_refinement,
        _format_meeting_context_for_llm,
    )
    from alteris_listener.llm.client import LLMClient
    from rich.console import Console

    console = Console()
    t0 = time.time()
    profile = _load_user_profile()
    user_tz = profile.get("timezone", "America/Los_Angeles")

    store = GraphStore()
    derived = DerivedStore()
    llm = LLMClient(provider="gemini", model="gemini-3-flash-preview",
                     thinking_level="medium")

    console.print(f"  [dim]Pipeline: finding events...[/dim]")
    events = _get_upcoming_events(store, days_ahead=days, user_tz=user_tz)

    if not events:
        console.print("  [yellow]No upcoming events.[/yellow]")
        store.close()
        derived.close()
        return {"briefing_ready": False, "voice_summary": "No upcoming events.",
                "open_items": []}

    console.print(f"  [dim]Pipeline: {len(events)} events[/dim]")

    # Gather context per event
    events_with_context = []
    for event in events:
        cls = event["_classification"]
        subj = event.get("subject", "(untitled)")

        if cls == "holiday":
            console.print(f"  [dim]Pipeline: holiday — {subj}[/dim]")
            ctx = {"related_emails": [], "related_meetings": [],
                   "related_messages": [], "commitments": [],
                   "contact_nodes": [], "context_summary": "holiday"}
            ts = event.get("timestamp", 0)
            ws, we = ts - 10 * 86400, ts + 2 * 86400
            terms = list(set(re.findall(r"[a-zA-Z]{4,}", subj.lower())))
            terms += ["closed", "closure", "holiday"]
            seen = set()
            for t in terms[:5]:
                for r in store.conn.execute(
                    """SELECT id, node_type, timestamp, subject, sender,
                              body_preview FROM nodes
                       WHERE (LOWER(subject) LIKE ? OR LOWER(body_preview) LIKE ?)
                         AND node_type IN ('email','message')
                         AND timestamp >= ? AND timestamp <= ?
                       ORDER BY timestamp DESC LIMIT 5""",
                    (f"%{t}%", f"%{t}%", ws, we),
                ).fetchall():
                    n = dict(r)
                    if n["id"] not in seen:
                        seen.add(n["id"])
                        ds = ""
                        if n.get("timestamp"):
                            ds = datetime.fromtimestamp(n["timestamp"]).strftime("%Y-%m-%d %H:%M")
                        ctx["related_emails"].append({
                            "id": n["id"], "date": ds,
                            "sender": n.get("sender", ""),
                            "subject": n.get("subject", ""),
                            "body_preview": n.get("body_preview", ""),
                            "hop": 0,
                        })
            if ctx["related_emails"]:
                ctx["context_summary"] = f"holiday — {len(ctx['related_emails'])} msgs"
            events_with_context.append((event, ctx, None))
            continue

        console.print(f"  [dim]Pipeline: context for {subj}...[/dim]")
        caps = (6, 3, 4) if cls in ("personal", "personal_external") else (12, 6, 8)
        context = _gather_meeting_context(
            event, store, derived,
            max_emails=caps[0], max_meetings=caps[1], max_messages=caps[2],
            user_tz=user_tz,
        )

        if cls in ("professional_external", "professional_internal"):
            console.print(f"  [dim]Pipeline: triage {subj}...[/dim]")
            context = _agentic_context_refinement(
                event, context, store, derived, llm, user_tz=user_tz,
            )

        events_with_context.append((event, context, None))

    # Generate voice summary + open items via LLM
    console.print("  [dim]Pipeline: generating voice summary...[/dim]")
    all_formatted = []
    for event, context, web in events_with_context:
        formatted = _format_meeting_context_for_llm(event, context, web)
        # Strip RSVP status lines — they cause the voice model to
        # narrate "X declined" which is noisy and often stale
        import re as _re
        formatted = _re.sub(r'\*\*RSVP status:\*\*[^\n]*\n?', '', formatted)
        all_formatted.append(formatted)

    separator = "\n\n" + "=" * 40 + "\n\n"
    context_block = separator.join(all_formatted)

    voice_data = llm.run_json(
        VOICE_SUMMARY_SYSTEM,
        f"Today is {datetime.now().strftime('%A, %B %d, %Y')}.\n\n"
        f"{len(events)} upcoming events:\n\n{context_block}"
    )

    voice_summary = ""
    open_items = []
    if isinstance(voice_data, dict):
        voice_summary = voice_data.get("voice_summary", "")
        open_items = voice_data.get("open_items", [])

    elapsed = time.time() - t0
    console.print(f"  [dim]Pipeline: stage A done ({elapsed:.1f}s, "
                  f"{len(events)} events, {len(open_items)} open items)[/dim]")

    return {
        "events_with_context": events_with_context,
        "events": events,
        "voice_summary": voice_summary,
        "open_items": open_items,
        "llm": llm,
        "store": store,
        "derived": derived,
        "user_tz": user_tz,
        "briefing_ready": True,
    }


def _run_pipeline_stage_b(stage_a: dict, user_answers: str = "") -> str:
    """Ambient context + synthesis → briefing markdown."""
    import time
    from alteris_listener.graph.briefing import (
        _gather_ambient_context, _synthesize_briefing,
    )
    from rich.console import Console

    console = Console()
    t0 = time.time()

    events = stage_a["events"]
    llm = stage_a["llm"]
    store = stage_a["store"]
    derived = stage_a["derived"]
    user_tz = stage_a["user_tz"]

    console.print("  [dim]Pipeline: ambient + synthesis...[/dim]")
    ambient = _gather_ambient_context(
        events, store=store, web_search=False, user_tz=user_tz,
    )

    briefing_md = _synthesize_briefing(
        stage_a["events_with_context"], llm,
        ambient_context=ambient,
        user_context=user_answers,
    )

    # Store in DB
    event_ids = [e["id"] for e in events]
    derived.conn.execute(
        """INSERT INTO briefings (briefing_type, content, commitment_ids,
           prompt_version, model_used, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        ("meeting_voice", briefing_md, json.dumps(event_ids),
         "voice_v2", "gemini-3-flash-preview", int(datetime.now().timestamp())),
    )
    derived.conn.commit()

    elapsed = time.time() - t0
    console.print(f"  [dim]Pipeline: stage B done ({elapsed:.1f}s)[/dim]")

    store.close()
    derived.close()
    return briefing_md


# ══════════════════════════════════════════════════════════════════
# Voice session
# ══════════════════════════════════════════════════════════════════

class VoiceSession:

    def __init__(self, days: int = DEFAULT_DAYS, no_tools: bool = False,
                 model: str = None, vertex: bool = False,
                 project: str = None, location: str = None):
        self.days = days
        self.no_tools = no_tools
        self.vertex = vertex
        self.project = project
        self.location = location or "us-central1"
        self.tool_handler = ToolHandler()
        self._model_speaking = False

        if vertex:
            self.model = model or VERTEX_MODEL
        else:
            self.model = model or DEV_MODEL

    @staticmethod
    def _get_api_key() -> str:
        from alteris_listener.llm.client import _get_api_key
        key = _get_api_key("GEMINI_API_KEY", "gemini")
        if not key:
            raise ValueError(
                "GEMINI_API_KEY not found. Either:\n"
                "  • Run:  alteris-listener set-key gemini\n"
                "  • Or:   export GEMINI_API_KEY='your-key'"
            )
        return key

    def _build_client(self) -> genai.Client:
        if self.vertex and self.no_tools:
            return genai.Client(
                vertexai=True,
                project=self.project,
                location=self.location,
            )
        else:
            if self.vertex and not self.no_tools:
                print("  ⚠️  Vertex AI + tools bug: using Developer API.\n")
            api_key = self._get_api_key()
            os.environ["GOOGLE_API_KEY"] = api_key
            return genai.Client()

    async def run(self):
        import pyaudio

        client = self._build_client()
        model = self.model
        if self.vertex and not self.no_tools:
            model = DEV_MODEL

        pya = pyaudio.PyAudio()

        print("\n╔══════════════════════════════════════════╗")
        print("║     Alteris Voice Briefing Session       ║")
        print("║  Speak naturally. Say 'goodbye' to end.  ║")
        print("╚══════════════════════════════════════════╝")
        print(f"  Model: {model} | Days: {self.days}\n")

        output_stream = pya.open(
            format=8, channels=1,
            rate=RECEIVE_SAMPLE_RATE, output=True,
            frames_per_buffer=CHUNK_SIZE,
        )

        stop_event = asyncio.Event()

        # ── Shared state ──
        stage_a_result = {}
        stage_a_done = asyncio.Event()
        conversation_done = asyncio.Event()
        user_answers: list[str] = []
        briefing_container: list[str] = []
        briefing_ready = asyncio.Event()

        # ── Background pipeline ──
        async def run_pipeline():
            nonlocal stage_a_result

            result = await asyncio.to_thread(_run_pipeline_stage_a, self.days)
            stage_a_result.update(result)
            stage_a_done.set()

            if not result.get("briefing_ready"):
                briefing_container.append("No upcoming events found.")
                briefing_ready.set()
                return

            # Wait for conversation to finish
            await conversation_done.wait()

            # Build user context from decisions + transcriptions
            answers_parts = []
            for d in self.tool_handler.decisions:
                answers_parts.append(f"[{d['event']}] {d['decision']}")
            for a in user_answers:
                if a.strip():
                    answers_parts.append(a.strip())

            answers_text = ""
            if answers_parts:
                answers_text = ("## User-Provided Context\n"
                                + "\n".join(f"- {p}" for p in answers_parts))

            md = await asyncio.to_thread(
                _run_pipeline_stage_b, stage_a_result, answers_text
            )
            briefing_container.append(md)
            briefing_ready.set()

        # ── Mic streaming ──
        async def stream_mic(session):
            mic_info = pya.get_default_input_device_info()
            mic_stream = await asyncio.to_thread(
                pya.open, format=8, channels=1,
                rate=SEND_SAMPLE_RATE, input=True,
                input_device_index=int(mic_info["index"]),
                frames_per_buffer=CHUNK_SIZE,
            )
            try:
                while not stop_event.is_set():
                    data = await asyncio.to_thread(
                        mic_stream.read, CHUNK_SIZE,
                        exception_on_overflow=False,
                    )
                    # Always send audio — the Live API handles echo
                    # cancellation server-side, and continuous audio
                    # keeps the websocket alive (prevents keepalive
                    # ping timeouts)
                    await session.send_realtime_input(
                        audio=types.Blob(
                            data=data,
                            mime_type=f"audio/pcm;rate={SEND_SAMPLE_RATE}",
                        )
                    )
            except asyncio.CancelledError:
                pass
            finally:
                mic_stream.close()

        # ── Response handler ──
        async def handle_responses(session, state):
            state.setdefault("user_turns", [])
            state.setdefault("model_turns", [])

            try:
                async for chunk in session.receive():
                    if (chunk.server_content
                            and chunk.server_content.model_turn
                            and chunk.server_content.model_turn.parts):
                        self._model_speaking = True
                        for part in chunk.server_content.model_turn.parts:
                            if part.inline_data and part.inline_data.data:
                                await asyncio.to_thread(
                                    output_stream.write,
                                    part.inline_data.data,
                                )

                    if (chunk.server_content
                            and hasattr(chunk.server_content, 'input_transcription')
                            and chunk.server_content.input_transcription):
                        txt = chunk.server_content.input_transcription.text
                        if txt:
                            print(f"\n  🎤 You: {txt}", flush=True)
                            state["user_turns"].append(txt)

                    if (chunk.server_content
                            and hasattr(chunk.server_content, 'output_transcription')
                            and chunk.server_content.output_transcription):
                        txt = chunk.server_content.output_transcription.text
                        if txt:
                            print(f"  🔊 Alteris: {txt}", end="", flush=True)
                            state["model_turns"].append(txt)

                    if (chunk.server_content
                            and chunk.server_content.turn_complete):
                        self._model_speaking = False
                        print()

                    if chunk.tool_call is not None:
                        for fc in chunk.tool_call.function_calls:
                            print(f"\n  🔧 {fc.name}("
                                  f"{json.dumps(fc.args or {}, default=str)[:80]})")
                            result = await self.tool_handler.handle(
                                fc.name, fc.args or {}
                            )
                            print(f"  ✅ {len(json.dumps(result, default=str))} chars")
                            await session.send_tool_response(
                                function_responses=types.FunctionResponse(
                                    name=fc.name,
                                    response={"result": json.dumps(
                                        result, default=str)},
                                    id=fc.id,
                                )
                            )
            except asyncio.CancelledError:
                pass
            except Exception as e:
                if "1006" in str(e) or "1008" in str(e) or "1011" in str(e):
                    print(f"\n  ⚠️  Connection closed: {e}")
                else:
                    print(f"\n  ❌ Response handler error: {e}")
        async def run_voice():
            # ═══ Session 1: Greeting (plays while pipeline runs) ═══
            greeting_system = GREETING_PROMPT.format(days=self.days)
            greeting_config: dict[str, Any] = {
                "response_modalities": ["AUDIO"],
                "system_instruction": greeting_system,
                "output_audio_transcription": {},
            }

            print("  🔊 Playing greeting...")
            async with client.aio.live.connect(
                model=model, config=greeting_config
            ) as greeting_session:

                # Trigger greeting
                await greeting_session.send_client_content(
                    turns=types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=(
                            f"Today is {datetime.now().strftime('%A, %B %d, %Y')}. "
                            f"Start my briefing session."
                        ))],
                    ),
                    turn_complete=True,
                )

                # Play greeting audio until turn completes
                async for chunk in greeting_session.receive():
                    if (chunk.server_content
                            and chunk.server_content.model_turn
                            and chunk.server_content.model_turn.parts):
                        for part in chunk.server_content.model_turn.parts:
                            if part.inline_data and part.inline_data.data:
                                await asyncio.to_thread(
                                    output_stream.write,
                                    part.inline_data.data,
                                )
                    if (chunk.server_content
                            and hasattr(chunk.server_content, 'output_transcription')
                            and chunk.server_content.output_transcription):
                        txt = chunk.server_content.output_transcription.text
                        if txt:
                            print(f"  🔊 Alteris: {txt}", end="", flush=True)
                    if (chunk.server_content
                            and chunk.server_content.turn_complete):
                        print()
                        break

            # Wait for pipeline if not done yet
            if not stage_a_done.is_set():
                print("  ⏳ Waiting for analysis to finish...")
            await stage_a_done.wait()

            voice_summary = stage_a_result.get("voice_summary", "")
            open_items = stage_a_result.get("open_items", [])

            # Format open items
            items_text = ""
            if open_items:
                parts = []
                for i, item in enumerate(open_items, 1):
                    parts.append(
                        f"Item {i}: {item.get('event', 'General')}\n"
                        f"Context: {item.get('context', '')}\n"
                        f"Question: {item.get('question', '')}\n"
                        f"If answered: {item.get('suggested_action', 'note for briefing')}"
                    )
                items_text = "\n\n".join(parts)
            else:
                items_text = "No open items — just give a warm overview of the week and ask if the user wants to dig into anything."

            # ═══ Session 2: Conversation (open items + follow-ups) ═══
            conversation_system = CONVERSATION_PROMPT.format(
                voice_summary=voice_summary,
                open_items=items_text,
            )

            conv_config: dict[str, Any] = {
                "response_modalities": ["AUDIO"],
                "system_instruction": conversation_system,
                "input_audio_transcription": {},
                "output_audio_transcription": {},
            }
            if not self.no_tools:
                conv_config["tools"] = [{"function_declarations": TOOL_DECLARATIONS}]

            # Track conversation state across reconnections
            all_state: dict[str, list] = {"user_turns": [], "model_turns": []}
            max_reconnects = 5
            reconnect_count = 0
            session_done = False

            while not session_done and reconnect_count <= max_reconnects:
                if reconnect_count > 0:
                    print(f"  🔄 Reconnecting (attempt {reconnect_count}/{max_reconnects})...")
                    await asyncio.sleep(1)
                else:
                    print("  🔊 Starting conversation...")

                try:
                    async with client.aio.live.connect(
                        model=model, config=conv_config
                    ) as session:

                        state: dict[str, list] = {"user_turns": [], "model_turns": []}
                        mic_task = asyncio.create_task(stream_mic(session))
                        resp_task = asyncio.create_task(
                            handle_responses(session, state)
                        )

                        # Build conversation seed
                        if reconnect_count == 0:
                            # First connection: fresh start
                            seed_turns = [
                                types.Content(
                                    role="user",
                                    parts=[types.Part.from_text(text=(
                                        f"Today is {datetime.now().strftime('%A, %B %d, %Y')}. "
                                        "I'm ready for my briefing."
                                    ))],
                                ),
                                types.Content(
                                    role="model",
                                    parts=[types.Part.from_text(text=(
                                        "Great, I've gone through everything. "
                                        "Let me walk you through what I found."
                                    ))],
                                ),
                                types.Content(
                                    role="user",
                                    parts=[types.Part.from_text(text="Go ahead.")],
                                ),
                            ]
                        else:
                            # Reconnection: seed with summary of what happened
                            covered = " ".join(all_state["model_turns"][-500:])
                            seed_turns = [
                                types.Content(
                                    role="user",
                                    parts=[types.Part.from_text(text=(
                                        "We were in the middle of my briefing but "
                                        "got disconnected. Continue where you left "
                                        "off. Here's what you already covered:\n\n"
                                        f"{covered[:2000]}"
                                    ))],
                                ),
                            ]

                        await session.send_client_content(
                            turns=seed_turns,
                            turn_complete=True,
                        )

                        # Run until goodbye or disconnect
                        max_session = 600
                        waited = 0
                        while waited < max_session and not stop_event.is_set():
                            await asyncio.sleep(2)
                            waited += 2

                            # Check for goodbye
                            combined_user = (
                                " ".join(all_state["user_turns"])
                                + " " + " ".join(state.get("user_turns", []))
                            ).lower()
                            if any(p in combined_user for p in
                                   ["goodbye", "that's it", "save the briefing",
                                    "save the pdf", "we're done", "that's all",
                                    "i'm good", "wrap up"]):
                                print("  ✅ User signaled end of session")
                                session_done = True
                                break

                        # Accumulate state
                        all_state["user_turns"].extend(state.get("user_turns", []))
                        all_state["model_turns"].extend(state.get("model_turns", []))

                        if session_done or stop_event.is_set():
                            # Goodbye
                            try:
                                await session.send_client_content(
                                    turns=types.Content(
                                        role="user",
                                        parts=[types.Part.from_text(text=(
                                            "I'm done. Say a brief warm goodbye "
                                            "and tell me you're saving the full "
                                            "briefing to my desktop."
                                        ))],
                                    ),
                                    turn_complete=True,
                                )
                                await asyncio.sleep(5)
                            except Exception:
                                pass
                            session_done = True

                        mic_task.cancel()
                        resp_task.cancel()
                        try:
                            await mic_task
                        except asyncio.CancelledError:
                            pass
                        try:
                            await resp_task
                        except asyncio.CancelledError:
                            pass

                except Exception as e:
                    err = str(e)
                    if "1006" in err or "1008" in err or "1011" in err:
                        print(f"  ⚠️  Connection dropped: {e}")
                        reconnect_count += 1
                        if reconnect_count > max_reconnects:
                            print("  ❌ Max reconnects reached, ending session")
                            session_done = True
                    else:
                        print(f"  ❌ Error: {e}")
                        traceback.print_exc()
                        session_done = True

            # Collect all transcriptions
            user_answers.extend(all_state.get("user_turns", []))

            # Signal pipeline to synthesize
            conversation_done.set()

            # Wait for briefing synthesis
            print("  ⏳ Synthesizing final briefing...")
            await briefing_ready.wait()
            print("  ✅ Briefing saved")

        # ── Run everything ──
        try:
            await asyncio.gather(run_pipeline(), run_voice())
        except asyncio.CancelledError:
            pass
        except Exception as e:
            err_str = str(e)
            if "1008" in err_str or "1011" in err_str:
                print(f"\n\n⚠️  Connection error: {e}")
            else:
                print(f"\n\n❌ Error: {e}")
                traceback.print_exc()
        finally:
            stop_event.set()
            output_stream.close()
            pya.terminate()
            print("\n✓ Voice session ended.")
            if self.tool_handler.decisions:
                print(f"  {len(self.tool_handler.decisions)} decision(s) recorded.")

            if briefing_container:
                try:
                    from alteris_listener.graph.briefing_pdf import render_briefing_pdf
                    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
                    pdf_path = Path.home() / "Desktop" / f"alteris_briefing_{ts}.pdf"
                    render_briefing_pdf(
                        briefing_container[0], pdf_path,
                        title="Meeting Briefing",
                    )
                    md_path = pdf_path.with_suffix(".md")
                    md_path.write_text(briefing_container[0])
                    print(f"  📄 Saved to {pdf_path}")
                except Exception as e:
                    print(f"  ⚠️  Could not save PDF: {e}")


# ══════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════

def run_voice_session(days: int = DEFAULT_DAYS, thinking: str = "medium",
                      model: str = None, no_tools: bool = False,
                      vertex: bool = False, project: str = None,
                      location: str = None):
    session = VoiceSession(
        days=days, model=model, no_tools=no_tools,
        vertex=vertex, project=project, location=location,
    )
    try:
        asyncio.run(session.run())
    except KeyboardInterrupt:
        print("\n\n✓ Session ended by user.")
