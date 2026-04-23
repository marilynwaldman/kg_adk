#  source .venv/bin/activate

```
uv sync                                                                                                                   
  uv run agent.py --file sample.txt 
                                                                                                                  

docker run -d --name falkordb -p 6379:6379 falkordb/falkordb:latest
docker run -d --name falkordb -p 6379:6379 -p 3000:3000 falkordb/falkordb:latest
```
                                                                                                                            
  That's it. Flags:                                                                                                         
  - -d — run in background                                                                                                  
  - --name falkordb — easy to reference later                                                                               
  - -p 6379:6379 — exposes the Redis-compatible port your agent connects to                                                 
                                       


# Knowledge Graph Extraction Agent

An ADK-style agent that reads a text file and extracts **entities** and **relationships** for insertion into a knowledge graph (Neo4j, AWS Neptune, or any property-graph database).

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                 KGExtractionAgent                   │
│                                                     │
│  User input: file path                              │
│       │                                             │
│       ▼                                             │
│  ┌──────────┐   ┌─────────────────┐                │
│  │read_file │──▶│extract_entities │                │
│  └──────────┘   └────────┬────────┘                │
│                           │                         │
│                           ▼                         │
│               ┌────────────────────────┐            │
│               │ extract_relationships  │            │
│               └───────────┬────────────┘            │
│                            │                        │
│                            ▼                        │
│                ┌──────────────────────┐             │
│                │  build_kg_payload    │             │
│                └──────────┬───────────┘             │
│                            │                        │
│                            ▼                        │
│              JSON payload  +  Cypher statements     │
└─────────────────────────────────────────────────────┘
```

### Tools

| Tool | Description |
|------|-------------|
| `read_file` | Reads the input `.txt` file from disk |
| `extract_entities` | Calls Claude to identify PERSON, ORG, LOCATION, EVENT, CONCEPT, PRODUCT, DATE nodes |
| `extract_relationships` | Calls Claude to find typed, directed edges between entities |
| `build_kg_payload` | Assembles nodes + edges + auto-generated Cypher `MERGE` statements |

---

## Quickstart

### 1. Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone and set up the project

```bash
git clone <repo-url>
cd kg-agent

# Install all dependencies into an isolated .venv (created automatically)
uv sync
```

### 3. Configure your API key

```bash
cp .env.example .env
# Open .env and set ANTHROPIC_API_KEY=sk-ant-...
```

`.env` is listed in `.gitignore` and will never be committed.

### 4. Run the agent

```bash
# Print the graph payload to stdout
uv run agent.py --file sample.txt

# Save the result to a JSON file
uv run agent.py --file sample.txt --output graph.json
```

---

## Environment Variables

All configuration lives in `.env` (copied from `.env.example`).

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ | — | Your Anthropic API key |
| `ANTHROPIC_MODEL` | ❌ | `claude-sonnet-4-20250514` | Model override |
| `ANTHROPIC_MAX_TOKENS` | ❌ | `2048` | Token limit per Claude call |

The agent loads `.env` automatically on startup via `python-dotenv`.  
You can also export variables in your shell — shell values take precedence over `.env`.

---

## Optional: Google ADK

To use the native ADK runner instead of the built-in ReAct loop:

```bash
uv sync --extra adk
```

The agent detects `google-adk` automatically and uses it when available; no code changes needed.

---

## Output Format

```json
{
  "metadata": {
    "source_file": "sample.txt",
    "node_count": 12,
    "edge_count": 18
  },
  "nodes": [
    {
      "id": "sam_altman",
      "label": "PERSON",
      "properties": {
        "name": "Sam Altman",
        "description": "CEO of OpenAI",
        "source": "sample.txt"
      }
    }
  ],
  "edges": [
    {
      "source": "sam_altman",
      "target": "openai",
      "type": "CEO_OF",
      "properties": {
        "description": "Sam Altman serves as the CEO of OpenAI.",
        "confidence": 1.0,
        "source": "sample.txt"
      }
    }
  ],
  "cypher": "MERGE (n:PERSON {id: \"sam_altman\"}) SET n += {...};\n..."
}
```

### Inserting into Neo4j

```python
from neo4j import GraphDatabase
import json

payload = json.load(open("graph.json"))
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

with driver.session() as session:
    for statement in payload["cypher"].split(";"):
        s = statement.strip()
        if s and not s.startswith("//"):
            session.run(s)
```

---

## Programmatic Usage

```python
from agent import run_agent

payload = run_agent("my_document.txt", output_path="graph.json")
print(payload["metadata"])
```

---

## Development

```bash
# Run linter
uv run ruff check .

# Run tests
uv run pytest

# Add a new dependency
uv add <package>

# Add a dev-only dependency
uv add --dev <package>
```

---

## Supported Entity Types

`PERSON` · `ORGANIZATION` · `LOCATION` · `EVENT` · `CONCEPT` · `PRODUCT` · `DATE` · `OTHER`

## Relationship Examples

`CEO_OF` · `FOUNDED_BY` · `INVESTED_IN` · `LOCATED_IN` · `WORKS_FOR` · `ACQUIRED_BY` · `PARTNER_OF` · `RELEASED`
