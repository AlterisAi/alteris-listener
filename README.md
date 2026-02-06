# alteris-listener

A CLI tool that reads from local macOS data sources — Mail.app, iMessage, Calendar, Slack, and [Granola](https://granola.ai) meeting transcripts — and runs LLM-powered queries against them to extract tasks, summaries, and insights.

Results can optionally be uploaded to the Alteris backend for use in the [Alteris](https://alteris.ai) AI Chief of Staff product.

## Requirements

- **macOS** (reads from macOS-native databases and Keychain)
- **Python 3.11+**
- **Full Disk Access** for your terminal app (required to read Mail.app and iMessage databases)
  - System Settings → Privacy & Security → Full Disk Access → add your terminal (Terminal.app, iTerm2, etc.)

## Installation

```bash
# Clone the repo
git clone https://github.com/AlterisAi/alteris-listener.git
cd alteris-listener

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with all LLM providers
pip install -e ".[all]"

# Or install with only the provider you need
pip install -e ".[gemini]"   # Google Gemini only
pip install -e ".[claude]"   # Anthropic Claude only
```

## Setup

### 1. Get an LLM API key

You need at least one LLM provider configured.

**Gemini (default, recommended to start):**

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Click "Create API Key"
3. Store it:

```bash
alteris-listener set-key gemini
# Paste your key when prompted
```

**Claude (alternative):**

1. Go to [Anthropic Console](https://console.anthropic.com/settings/keys)
2. Create an API key
3. Store it:

```bash
alteris-listener set-key claude
# Paste your key when prompted
```

Keys are stored securely in macOS Keychain. You can also set them via environment variables (`GEMINI_API_KEY` or `ANTHROPIC_API_KEY`).

### 2. Set up your queries directory

Queries are markdown files with YAML frontmatter that define the prompts sent to the LLM. You need to tell the CLI where to find them.

**Option A: Environment variable (recommended)**

```bash
# Add to your ~/.zshrc or ~/.bashrc
export ALTERIS_QUERIES_DIR=/path/to/your/queries
```

**Option B: CLI flag**

```bash
alteris-listener run-query my_query -s mail --queries-dir /path/to/your/queries
```

**Option C: Default**

If neither is set, the CLI looks for a `queries/` directory in your current working directory.

### 3. Create a query file

Create a `.md` file in your queries directory. Example `todo_extractor.md`:

```markdown
---
query_name: todo_extractor
sources: [mail, granola]
description: Extract action items from emails or meetings
---

You are a task extraction assistant. Given the following content,
extract all action items, todos, and commitments.

For each task, return JSON with:
- title: short description
- assignee: who is responsible
- priority: 1 (high), 2 (medium), or 3 (low)
- due_date: ISO date if mentioned, null otherwise
- done: false

Return a JSON object: {"tasks": [...]}
```

The YAML frontmatter defines metadata. Everything after the second `---` is the prompt sent to the LLM, with the source content appended as context.

### 4. Grant Full Disk Access (for Mail and iMessage)

If you want to query Mail.app or iMessage:

1. Open **System Settings → Privacy & Security → Full Disk Access**
2. Click the **+** button and add your terminal app
3. Restart your terminal

### 5. Slack setup (optional)

To read from Slack, you need a bot token:

1. Create a Slack app at [api.slack.com/apps](https://api.slack.com/apps)
2. Add OAuth scopes: `channels:history`, `channels:read`, `groups:history`, `groups:read`, `users:read`
3. Install to your workspace and copy the Bot User OAuth Token (`xoxb-...`)
4. Store it:

```bash
alteris-listener set-key slack
```

### 6. Granola setup (optional)

[Granola](https://granola.ai) is a meeting transcription app. If you have it installed and logged in, `alteris-listener` can read your meetings automatically — no additional setup needed. It reads auth tokens from Granola's local storage.

## Usage

### Run queries against a data source

```bash
# Extract todos from recent emails (last 24 hours)
alteris-listener run-query todo_extractor -s mail

# Summarize meetings from the last week
alteris-listener run-query meeting_summary -s granola --hours 168

# Run multiple queries per item
alteris-listener run-query meeting_summary meeting_todo_extractor -s granola --hours 168

# Process more items
alteris-listener run-query todo_extractor -s mail --hours 48 --max 20

# Use Claude instead of Gemini
alteris-listener run-query meeting_summary -s granola --provider claude

# Print raw JSON output
alteris-listener run-query todo_extractor -s mail --raw
```

### Ask a freeform question

```bash
alteris-listener ask "What meetings do I have tomorrow?"
alteris-listener ask "Summarize my unread emails from today"
alteris-listener ask "What happened in Slack today?" --source slack
alteris-listener ask "What was discussed in my last meeting?" --source granola
```

### Upload results to Alteris

If you have an Alteris account, you can sync results to the cloud:

```bash
# Log in first (opens browser for Google sign-in)
alteris-listener login

# Check login status
alteris-listener whoami

# Run queries and upload results
alteris-listener run-query meeting_summary meeting_todo_extractor -s granola --upload

# Inject user context (profile, goals) from the cloud into LLM calls
alteris-listener run-query meeting_summary -s granola -c clarity_queue -c goals_and_values --upload

# Skip cloud context
alteris-listener run-query meeting_summary -s granola --no-context

# Log out
alteris-listener logout
```

## Data sources

| Source | What it reads | Requirements |
|--------|--------------|--------------|
| `mail` | Mail.app emails via SQLite | Full Disk Access |
| `imessage` | iMessage via SQLite | Full Disk Access |
| `calendar` | macOS Calendar via EventKit | Calendar permission grant |
| `slack` | Slack channels via API | Bot token (`set-key slack`) |
| `granola` | Meeting transcripts via Granola API | Granola app installed & logged in |

## Command reference

| Command | Description |
|---------|-------------|
| `run-query QUERY [QUERY...] -s SOURCE` | Run queries against a data source |
| `ask "question"` | Ask a freeform question about your data |
| `set-key PROVIDER` | Store an API key in Keychain (gemini, claude, slack) |
| `login` | Log in to Alteris (opens browser) |
| `logout` | Log out and remove stored credentials |
| `whoami` | Check current login status |

### run-query options

| Option | Default | Description |
|--------|---------|-------------|
| `--source`, `-s` | *required* | Data source: mail, imessage, calendar, slack, granola |
| `--queries-dir` | `$ALTERIS_QUERIES_DIR` or `./queries/` | Path to query markdown files |
| `--provider` | gemini | LLM provider: gemini or claude |
| `--model` | auto | Override model name |
| `--thinking` | low | Thinking budget: off, minimal, low, medium, high |
| `--hours` | 24 | Hours of history to look back |
| `--max` | 10 | Max items to process |
| `--thread-id` | — | Process a specific email thread (mail only) |
| `--user-email` | — | Your email address (auto-detected from cloud context) |
| `--raw` | off | Print raw JSON output |
| `--upload` | off | Upload results to Alteris |
| `--context`, `-c` | clarity_queue | Cloud context docs to inject into LLM calls |
| `--no-context` | off | Skip fetching cloud context |

## Project structure

```
alteris_listener/
├── cli/              # CLI commands
│   ├── query_cmd.py      run-query command
│   ├── ask_cmd.py        ask command
│   └── auth_cmd.py       login/logout/whoami
├── sources/          # Data source readers
│   ├── base.py           Message dataclass
│   ├── mail.py           Mail.app (SQLite)
│   ├── imessage.py       iMessage (SQLite)
│   ├── calendar.py       Calendar (EventKit)
│   ├── slack.py          Slack (API)
│   └── granola.py        Granola (API)
├── llm/              # LLM interaction
│   ├── client.py         Gemini/Claude wrapper
│   ├── context.py        Format messages for LLM prompts
│   └── loader.py         Load query definitions from markdown
├── api/              # Alteris backend communication
│   ├── config.py         API config and Keychain helpers
│   ├── session.py        Auth session management
│   ├── upload.py         Upload results
│   └── context.py        Fetch user context from cloud
└── main.py           # CLI entry point
```

## License

MIT
