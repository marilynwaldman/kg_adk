"""
Knowledge Graph Extraction Agent  (ADK-only)
=============================================
Reads a text file and extracts entities + relationships for a knowledge graph.

Tools
-----
  read_file              — reads the input .txt file from disk
  extract_entities       — uses the ADK agent's LLM to identify named entities
  extract_relationships  — uses the ADK agent's LLM to find typed relationships
  build_kg_payload       — assembles nodes + edges + Cypher into a final payload

Setup
-----
    cp .env.example .env       # set ANTHROPIC_API_KEY
    uv sync --extra adk
    uv run agent.py --file sample.txt
    uv run agent.py --file sample.txt --output graph.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# ADK imports  (hard requirement — install with: uv sync --extra adk)
# ---------------------------------------------------------------------------
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai.types import Content, Part

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL = LiteLlm(model="anthropic/claude-sonnet-4-20250514")

SYSTEM_PROMPT = """\
You are a Knowledge Graph Extraction Agent.

Your goal is to build a complete knowledge-graph payload from a text file.
Work through these steps IN ORDER, calling one tool at a time:

  1. read_file           — read the text file
  2. extract_entities    — extract all named entities from the text
  3. extract_relationships — extract all relationships between those entities
  4. build_kg_payload    — assemble the final graph payload

Rules:
- Pass the raw text content (not the file path) to extract_entities.
- Pass the raw text AND the full entity list to extract_relationships.
- Pass the entity list, relationship list, and source file path to build_kg_payload.
- After build_kg_payload completes, your job is done. Stop calling tools.
"""

# ---------------------------------------------------------------------------
# JSON parse helper
# ---------------------------------------------------------------------------

def _parse_json(raw: str, array_key: str) -> dict[str, Any]:
    """Parse a JSON blob that may be wrapped in markdown fences or truncated."""
    text = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    text = re.sub(r"\s*```$", "", text)

    # 1 — clean parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2 — outermost { … }
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e > s:
        try:
            return json.loads(text[s : e + 1])
        except json.JSONDecodeError:
            pass

    # 3 — salvage complete objects from a truncated array
    objects: list[dict] = []
    depth, obj_start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start is not None:
                try:
                    objects.append(json.loads(text[obj_start : i + 1]))
                except json.JSONDecodeError:
                    pass
                obj_start = None

    if objects:
        if array_key in objects[0]:
            return objects[0]
        return {array_key: objects[1:] if len(objects) > 1 else objects}

    return {"error": "Could not parse JSON", "raw": raw[:500]}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def read_file(file_path: str) -> dict[str, Any]:
    """Read a plain-text file and return its full contents.

    Args:
        file_path: Path to the .txt file to read.

    Returns:
        dict with keys 'content' (str) and 'char_count' (int), or 'error' (str).
    """
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        return {"error": f"File not found: {path}"}
    if not path.is_file():
        return {"error": f"Not a file: {path}"}
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        return {"content": content, "char_count": len(content)}
    except OSError as exc:
        return {"error": str(exc)}


def extract_entities(text: str) -> dict[str, Any]:
    """Extract all named entities from raw text.

    Uses a dedicated LLM call to identify people, organisations, locations,
    events, concepts, products, dates, and other named entities.

    Args:
        text: The raw text content to analyse.

    Returns:
        dict with key 'entities' (list of dicts) or 'error' (str).
        Each entity has: id, name, type, description.
    """
    # We make a direct LiteLLM call here so entity extraction is its own
    # focused prompt, isolated from the agent's orchestration context.
    import litellm  # installed as part of the adk extra

    prompt = f"""\
Extract every named entity from the text below.

Return a JSON object with a single key "entities" whose value is an array.
Each element must have:
  "id"          : short snake_case identifier unique in this document
  "name"        : canonical display name
  "type"        : one of PERSON | ORGANIZATION | LOCATION | EVENT | CONCEPT | PRODUCT | DATE | OTHER
  "description" : one-sentence description from context (or "" if unclear)

Return ONLY valid JSON — no markdown fences, no commentary.

TEXT:
\"\"\"
{text[:8000]}
\"\"\"
"""
    response = litellm.completion(
        model="anthropic/claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
    )
    raw = response.choices[0].message.content or ""
    return _parse_json(raw, "entities")


def extract_relationships(text: str, entities: list) -> dict[str, Any]:
    """Extract relationships between named entities found in raw text.

    Args:
        text:     The raw text content to analyse.
        entities: List of entity dicts returned by extract_entities.

    Returns:
        dict with key 'relationships' (list of dicts) or 'error' (str).
        Each relationship has: source_id, target_id, relation, description, confidence.
    """
    import litellm

    entity_summary = json.dumps(
        [{"id": e["id"], "name": e["name"], "type": e["type"]} for e in entities],
        indent=2,
    )

    prompt = f"""\
Given the entities and source text below, extract every meaningful relationship
between any two entities.

ENTITIES:
{entity_summary}

Return a JSON object with key "relationships", an array where each element has:
  "source_id"   : id of the subject entity
  "target_id"   : id of the object entity
  "relation"    : concise UPPER_SNAKE_CASE predicate (e.g. WORKS_FOR, FOUNDED_BY)
  "description" : one sentence describing the relationship
  "confidence"  : float 0.0-1.0 — how explicitly stated the relationship is

Return ONLY valid JSON — no markdown fences, no commentary.

TEXT:
\"\"\"
{text[:8000]}
\"\"\"
"""
    response = litellm.completion(
        model="anthropic/claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
    )
    raw = response.choices[0].message.content or ""
    return _parse_json(raw, "relationships")


def build_kg_payload(
    entities: list,
    relationships: list,
    source_file: str = "",
) -> dict[str, Any]:
    """Assemble a knowledge-graph payload ready for database ingestion.

    Produces nodes, edges, and auto-generated Neo4j Cypher MERGE statements.

    Args:
        entities:      List of entity dicts from extract_entities.
        relationships: List of relationship dicts from extract_relationships.
        source_file:   Original file path, stored as a property on every node/edge.

    Returns:
        dict with keys: metadata, nodes, edges, cypher.
    """
    nodes = [
        {
            "id": e.get("id", ""),
            "label": e.get("type", "Entity"),
            "properties": {
                "name": e.get("name", ""),
                "description": e.get("description", ""),
                "source": source_file,
            },
        }
        for e in entities
    ]

    edges = [
        {
            "source": r.get("source_id", ""),
            "target": r.get("target_id", ""),
            "type": r.get("relation", "RELATED_TO"),
            "properties": {
                "description": r.get("description", ""),
                "confidence": r.get("confidence", 1.0),
                "source": source_file,
            },
        }
        for r in relationships
    ]

    cypher_lines = [
        "// Auto-generated Cypher — paste into Neo4j Browser or use neo4j-driver\n"
    ]
    for n in nodes:
        props = ", ".join(
            f"{k}: {json.dumps(v)}" for k, v in n["properties"].items()
        )
        cypher_lines.append(
            f'MERGE (n:{n["label"]} {{id: {json.dumps(n["id"])}}}) SET n += {{{props}}};'
        )
    cypher_lines.append("")
    for edge in edges:
        props = ", ".join(
            f"{k}: {json.dumps(v)}" for k, v in edge["properties"].items()
        )
        cypher_lines.append(
            f'MATCH (a {{id: {json.dumps(edge["source"])}}}), '
            f'(b {{id: {json.dumps(edge["target"])}}})\n'
            f'MERGE (a)-[r:{edge["type"]}]->(b) SET r += {{{props}}};'
        )

    return {
        "metadata": {
            "source_file": source_file,
            "node_count": len(nodes),
            "edge_count": len(edges),
        },
        "nodes": nodes,
        "edges": edges,
        "cypher": "\n".join(cypher_lines),
    }


# ---------------------------------------------------------------------------
# Agent + Runner
# ---------------------------------------------------------------------------

def _build_agent() -> Agent:
    return Agent(
        name="kg_extraction_agent",
        model=MODEL,
        description="Reads a text file and extracts entities and relationships for a knowledge graph.",
        instruction=SYSTEM_PROMPT,
        tools=[
            FunctionTool(read_file),
            FunctionTool(extract_entities),
            FunctionTool(extract_relationships),
            FunctionTool(build_kg_payload),
        ],
    )


async def _run_async(file_path: str) -> dict[str, Any]:
    """Run the agent and capture the build_kg_payload result directly."""
    agent = _build_agent()
    session_svc = InMemorySessionService()
    runner = Runner(agent=agent, app_name="kg_agent", session_service=session_svc)
    session = await session_svc.create_session(app_name="kg_agent", user_id="user")

    user_msg = Content(
        role="user",
        parts=[Part(text=f"Process file: {file_path}")],
    )

    captured: dict[str, Any] = {}

    async for event in runner.run_async(
        user_id="user",
        session_id=session.id,
        new_message=user_msg,
    ):
        # ADK emits events for every tool call and response.
        # Intercept the build_kg_payload tool result directly from the event
        # stream — never rely on the agent's final text response.
        if hasattr(event, "content") and event.content:
            for part in event.content.parts or []:
                if hasattr(part, "function_response") and part.function_response:
                    fn = part.function_response
                    if fn.name == "build_kg_payload":
                        response_data = fn.response
                        # ADK wraps tool return values in {"result": ...}
                        payload = response_data.get("result", response_data)
                        if isinstance(payload, dict) and "nodes" in payload:
                            captured = payload
                            print(
                                f"  ✅  Captured payload: "
                                f"{payload['metadata']['node_count']} nodes, "
                                f"{payload['metadata']['edge_count']} edges"
                            )

    return captured


# ---------------------------------------------------------------------------
# Public API + CLI
# ---------------------------------------------------------------------------

def run_agent(file_path: str, output_path: str | None = None) -> dict[str, Any]:
    """Run the KG extraction agent on *file_path* and optionally save the result.

    Args:
        file_path:   Path to the input text file.
        output_path: If given, write the JSON payload here.

    Returns:
        The knowledge-graph payload dict.
    """
    print(f"🤖  KG Extraction Agent starting\n    file: {file_path}\n")
    payload = asyncio.run(_run_async(file_path))

    if not payload:
        print("⚠️   No payload captured — check for tool errors above.")
        return {}

    if output_path:
        Path(output_path).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n✅  Result saved to: {output_path}")
    else:
        print("\n── KNOWLEDGE GRAPH PAYLOAD ──────────────────────────────────────")
        print(json.dumps(payload, indent=2, ensure_ascii=False))

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ADK agent: extract entities & relationships from a text file."
    )
    parser.add_argument("--file", required=True, help="Path to the input .txt file")
    parser.add_argument("--output", default=None, help="Write JSON result to this file")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit(
            "Error: ANTHROPIC_API_KEY is not set.\n"
            "Add it to your .env file or export it in your shell."
        )

    run_agent(args.file, args.output)


if __name__ == "__main__":
    main()
