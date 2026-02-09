# alteris-listener

A local-first AI chief of staff that reads your macOS data sources — Mail.app, iMessage, Calendar, Slack, and [Granola](https://granola.ai) meeting transcripts — builds a personal knowledge graph, and generates actionable briefings for your upcoming week.

Everything runs on your machine. Your data never leaves your laptop unless you explicitly enable cloud features.

## What it does

**Morning briefing example** — you run `alteris-listener graph brief --web --ask` and get:

- **Team call**: "Lead with the local-first pivot. Dashboards are handed off to Vic and Sean. Ahmad still needs Analytics access."
- **Valentine's Day**: "Ananya has a Valentine party at school Friday (per Konstella email). Flowers: Trader Joe's ($5) or Slow Fox ($30+)."
- **Birthday party → dinner**: "Party ends 3:30 in Wallingford. Dinner at 4:00 in Magnolia. Leave by 3:30 sharp — Luci arrives at 3:30 for childcare handoff."
- **Presidents' Day**: "⚠️ School and daycare closed. No childcare arranged. Draft message to Flor: 'Hi Flor, schools are closed Monday...'"

It connects dots across your email, messages, calendar, and past meetings that you'd otherwise miss.

## Architecture

The system builds a local knowledge graph from your data, then walks it to prepare context-rich briefings.

```
┌─────────────────────────────────────────────────────┐
│                    Data Sources                      │
│  Mail.app · iMessage · Calendar · Slack · Granola   │
└────────────────────────┬────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │   Pass 0: Ingest    │  Structural extraction
              │   Pass 1: Score     │  Heuristic ranking
              │   Pass 2: Embed     │  Semantic embeddings
              │   Pass 3: Triage    │  LLM-powered tier assignment
              │   Pass 4: Extract   │  Commitment extraction
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │   Knowledge Graph   │  graph.db (SQLite)
              │   Nodes + Edges     │  derived.db (commitments)
              │   Contact Stats     │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │   Pass 5: Brief     │  Multi-hop graph walk
              │   Triage → Refine   │  Agentic context gathering
              │   Questions → Synth │  Interactive Q&A (optional)
              └─────────────────────┘
```

### Graph passes

| Pass | Command | What it does | LLM? |
|------|---------|-------------|------|
| 0+1 | `graph bootstrap` | Ingest all sources, score nodes by heuristics | No |
| 2 | `graph embed` | Generate embeddings for semantic search | Local model |
| 3 | `graph triage` | LLM classifies nodes into importance tiers | Yes (Gemini) |
| — | `graph entity-edges` | Link nodes by shared entities and topics | Yes (Gemini) |
| — | `graph propagate` | Spread importance scores along edges | No |
| 4 | `graph extract` | Extract commitments/tasks from email threads | Yes (Gemini) |
| 5 | `graph brief` | Generate meeting briefings from the graph | Yes (Gemini) |

## Requirements

- **macOS** (reads from macOS-native databases and Keychain)
- **Python 3.11+** (3.13 recommended)
- **Full Disk Access** for your terminal app (required to read Mail.app and iMessage databases)
  - System Settings → Privacy & Security → Full Disk Access → add your terminal

## Installation

```bash
# Clone and set up
git clone https://github.com/AlterisAi/alteris-listener.git
cd alteris-listener
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[all]"

# Store your API key (Gemini recommended)
alteris-listener set-key gemini
```

Keys are stored in macOS Keychain. You can also use environment variables (`GEMINI_API_KEY` or `ANTHROPIC_API_KEY`).

## Quick start: build graph and get your first briefing

```bash
# 1. Build the knowledge graph from your local data
alteris-listener graph bootstrap

# 2. Generate embeddings (uses local model, no API key needed)
alteris-listener graph embed

# 3. Run LLM triage to classify nodes by importance
alteris-listener graph triage

# 4. Link nodes by shared entities and topics
alteris-listener graph entity-edges

# 5. Propagate importance scores along edges
alteris-listener graph propagate

# 6. Extract commitments from email threads
alteris-listener graph extract

# 7. Generate your briefing
alteris-listener graph brief --web --ask
```

Steps 1-6 build the graph (run once, then incrementally). Step 7 is the daily output.

### Briefing options

```bash
# Basic briefing (next 7 days, no web search)
alteris-listener graph brief

# With web search for attendee backgrounds and venue info
alteris-listener graph brief --web

# Interactive mode — agent asks you questions it can't answer from the graph
alteris-listener graph brief --web --ask

# Longer time window
alteris-listener graph brief --days 14 --web --ask

# Save to file
alteris-listener graph brief --web --save ~/Desktop/weekly-brief.md

# Use higher thinking for more nuanced synthesis
alteris-listener graph brief --web --ask --thinking high
```

### User profile

The briefing system reads `~/.alteris/profile.yaml` for personalization — home location, family, travel preferences, local knowledge. This is optional but makes briefings significantly better.

```yaml
name: "Jane Smith"
timezone: "America/New_York"
emails:
  - jane@company.com
  - janesmith@gmail.com
home:
  neighborhood: "Park Slope"
  city: "Brooklyn"
  state: "NY"
family:
  spouse: "Alex Smith"
  children:
    - name: "Sam"
      school: "PS 321"
  care_providers:
    - name: "Maria"
      role: "Babysitter, evenings"
travel:
  airline_loyalty: "Delta"
  seat_preference: "aisle"
local_knowledge:
  - "Parking on 5th Ave is terrible on weekends"
  - "Trader Joe's on Atlantic has the best flowers"
```

## Graph commands

```bash
# Build and maintain
alteris-listener graph bootstrap              # Ingest all sources
alteris-listener graph bootstrap -s mail      # Ingest mail only
alteris-listener graph embed                  # Generate embeddings
alteris-listener graph triage                 # LLM importance classification
alteris-listener graph entity-edges           # Link by entities/topics
alteris-listener graph propagate              # Score propagation
alteris-listener graph extract                # Commitment extraction

# Inspect
alteris-listener graph status                 # Graph stats and health
alteris-listener graph contacts               # Contact rankings
alteris-listener graph node <id>              # Inspect a specific node
alteris-listener graph neighbors <id>         # Show node neighborhood
alteris-listener graph scores                 # Score distribution analysis

# Output
alteris-listener graph brief --web --ask      # Meeting briefing

# Maintenance
alteris-listener graph reset                  # Wipe and rebuild
alteris-listener graph extract-reset          # Reset extraction state
alteris-listener graph dedup                  # Deduplicate contacts
```

## Query system

In addition to the graph, you can run structured queries against individual data sources:

```bash
# Run a query against a source
alteris-listener run-query todo_extractor -s mail
alteris-listener run-query meeting_summary -s granola --hours 168

# Ask a freeform question
alteris-listener ask "What meetings do I have tomorrow?"
alteris-listener ask "Summarize my unread emails from today"
```

Queries are markdown files with YAML frontmatter stored in a queries directory. Set `ALTERIS_QUERIES_DIR` or use `--queries-dir`.

## Data sources

| Source | What it reads | Requirements |
|--------|--------------|--------------|
| `mail` | Mail.app emails via SQLite | Full Disk Access |
| `imessage` | iMessage via SQLite | Full Disk Access |
| `calendar` | macOS Calendar via EventKit | Calendar permission |
| `slack` | Slack channels via API | Bot token (`set-key slack`) |
| `granola` | Meeting transcripts via Granola API | Granola installed |

## Data storage

All data stays local:

```
~/.alteris/
├── graph.db          # Knowledge graph (nodes, edges, contact stats)
├── derived.db        # Extracted commitments and tasks
├── profile.yaml      # User profile for personalization
└── embeddings/       # Cached semantic embeddings
```

## Project structure

```
alteris_listener/
├── cli/
│   ├── graph_cmd.py      # Graph commands (bootstrap, brief, etc.)
│   ├── query_cmd.py      # run-query command
│   ├── ask_cmd.py        # ask command
│   └── auth_cmd.py       # login/logout/whoami
├── graph/
│   ├── bootstrap.py      # Pass 0+1: ingest and score
│   ├── embeddings.py     # Pass 2: semantic embeddings
│   ├── triage.py         # Pass 3: LLM importance triage
│   ├── entity_edges.py   # Entity/topic edge creation
│   ├── propagate.py      # Score propagation along edges
│   ├── extract.py        # Pass 4: commitment extraction
│   ├── briefing.py       # Pass 5: meeting briefing synthesis
│   ├── store.py          # Graph database (SQLite)
│   ├── derived_store.py  # Derived data (commitments)
│   ├── schema.py         # Database schema
│   ├── scoring.py        # Heuristic scoring
│   ├── ingest.py         # Source-specific ingestion
│   ├── entities.py       # Entity extraction
│   ├── dedup.py          # Contact deduplication
│   ├── contacts_resolver.py  # Contact resolution
│   ├── email_cleaner.py  # Email body cleanup
│   ├── local_llm.py      # Local LLM interface
│   ├── neighbors.py      # Graph neighborhood queries
│   └── summarize.py      # Node summarization
├── sources/              # Data source readers
├── llm/                  # LLM client wrapper
├── api/                  # Alteris cloud backend (optional)
└── main.py               # CLI entry point
```

## License

MIT
