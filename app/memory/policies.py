MEMORY_EXTRACTION_PROMPT = """
You extract durable roleplaying memory from a completed turn.

Only write memory if it will matter later.

Write memory for:
- player decisions
- relationship changes
- quest changes
- promises
- threats
- discovered secrets
- durable world facts
- strong player preferences

Do not write memory for:
- greetings
- filler dialogue
- repeated facts
- short emotional reactions without consequence

Return exactly one JSON object and no markdown.
Use this shape:
{
  "write_memory": true | false,
  "memories": [
    {
      "summary": "...",
      "visibility": "player" | "gm" | "character_private",
      "importance": 1,
      "tags": ["..."],
      "scene_id": "..." | null,
      "actor_id": "..." | null
    }
  ],
  "reason": "..."
}
""".strip()
