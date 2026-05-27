# 03 — MVP Implementation Guide

## Purpose

This document describes how to implement the first working MVP of **RoleRAG_POC**.

The MVP is a personal-use Python roleplaying RAG system for one user. It must support:

- one local 8B-class model as the default model
- one optional cloud model fallback
- structured scenes and personas
- retrieval-augmented context assembly
- durable memory extraction
- a controlled multi-agent turn pipeline
- simple CLI usage first
- FastAPI later, but only after the core loop works

The goal is not to build a complete platform. The goal is to build a small, inspectable engine that proves the architecture works.

---

## Implementation Philosophy

The MVP must be boring, explicit, and testable.

Do not build a clever autonomous agent swarm. Build a deterministic Python application that calls LLMs for narrow tasks.

The Python code owns:

- state loading
- context budgeting
- visibility filtering
- retrieval filtering
- provider routing
- memory persistence
- retry limits
- validation boundaries

The LLM owns:

- text generation
- structured classification
- structured critique
- structured memory proposal
- summarization

The LLM must not own authoritative state.

---

## MVP Runtime Goal

The first complete MVP loop should work like this:

```text
User enters a roleplay message
  -> engine loads active session
  -> engine loads active scene
  -> engine loads active persona
  -> engine retrieves relevant memories/lore
  -> engine builds a compact context packet
  -> local model generates a response
  -> critic checks the response
  -> response is returned to the user
  -> memory curator proposes durable memory
  -> engine validates and stores memory
```

This is enough for a meaningful proof of concept.

---

## Initial Repository Structure

Create this structure first:

```text
RoleRAG_POC/
  docs/
    01_product_goal.md
    02_architecture.md
    03_implementation_guide.md

  app/
    __init__.py
    config.py
    main.py

    domain/
      __init__.py
      models.py
      visibility.py

    llm/
      __init__.py
      provider.py
      openai_compatible.py
      router.py

    orchestration/
      __init__.py
      turn_orchestrator.py
      context_builder.py
      context_budget.py

    agents/
      __init__.py
      intent_classifier.py
      actor_agent.py
      critic_agent.py
      memory_curator.py
      retrieval_agent.py

    rag/
      __init__.py
      embeddings.py
      ingestion.py
      vector_store.py
      retriever.py

    memory/
      __init__.py
      store.py
      summarizer.py

    persistence/
      __init__.py
      sqlite.py
      repositories.py

  data/
    worlds/
    personas/
    documents/
    sessions/

  tests/
    unit/
    integration/
    evals/

  .env.example
  .gitignore
  docker-compose.yml
  pyproject.toml
  README.md
```

Do not add unused folders beyond this during the MVP phase.

---

## Recommended Dependencies

Use a small Python stack.

```toml
[project]
name = "rolerag-poc"
version = "0.1.0"
description = "Personal roleplaying RAG proof of concept"
requires-python = ">=3.12"
dependencies = [
  "fastapi>=0.115.0",
  "uvicorn[standard]>=0.30.0",
  "typer>=0.12.0",
  "pydantic>=2.8.0",
  "pydantic-settings>=2.4.0",
  "openai>=1.40.0",
  "httpx>=0.27.0",
  "qdrant-client>=1.11.0",
  "sentence-transformers>=3.0.0",
  "sqlalchemy>=2.0.0",
  "aiosqlite>=0.20.0",
  "python-dotenv>=1.0.0",
  "rich>=13.7.0"
]

[project.optional-dependencies]
dev = [
  "pytest>=8.3.0",
  "pytest-asyncio>=0.23.0",
  "ruff>=0.6.0",
  "mypy>=1.11.0"
]
```

For a first pass, use `uv` or plain `pip`. Do not add Poetry-specific complexity unless needed.

---

## Configuration

Create `.env.example`:

```env
APP_ENV=local
DATABASE_URL=sqlite+aiosqlite:///./data/rolerag.db
QDRANT_URL=http://localhost:6333

LOCAL_LLM_BASE_URL=http://localhost:11434/v1
LOCAL_LLM_API_KEY=ollama
LOCAL_LLM_MODEL=qwen3:8b
LOCAL_LLM_TEMPERATURE=0.75
LOCAL_LLM_MAX_TOKENS=700

CLOUD_MODE=ask
CLOUD_LLM_BASE_URL=https://api.openai.com/v1
CLOUD_LLM_API_KEY=replace_me
CLOUD_LLM_MODEL=gpt-4.1-mini
CLOUD_LLM_TEMPERATURE=0.65
CLOUD_LLM_MAX_TOKENS=1000

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RETRIEVAL_TOP_K=6
RECENT_DIALOGUE_TURNS=8
MAX_LOCAL_RETRIES=1
MAX_TOTAL_AGENT_CALLS=4
```

Create `app/config.py` with typed settings:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_env: str = "local"
    database_url: str = "sqlite+aiosqlite:///./data/rolerag.db"
    qdrant_url: str = "http://localhost:6333"

    local_llm_base_url: str = "http://localhost:11434/v1"
    local_llm_api_key: str = "ollama"
    local_llm_model: str = "qwen3:8b"
    local_llm_temperature: float = 0.75
    local_llm_max_tokens: int = 700

    cloud_mode: str = "ask"
    cloud_llm_base_url: str = "https://api.openai.com/v1"
    cloud_llm_api_key: str = "replace_me"
    cloud_llm_model: str = "gpt-4.1-mini"
    cloud_llm_temperature: float = 0.65
    cloud_llm_max_tokens: int = 1000

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    retrieval_top_k: int = 6
    recent_dialogue_turns: int = 8
    max_local_retries: int = 1
    max_total_agent_calls: int = 4


def get_settings() -> Settings:
    return Settings()
```

---

## Domain Models

The MVP must start with structured data models. Do not begin with prompt strings only.

Create `app/domain/visibility.py`:

```python
from enum import StrEnum


class Visibility(StrEnum):
    PLAYER = "player"
    GM = "gm"
    CHARACTER_PRIVATE = "character_private"
```

Create `app/domain/models.py`:

```python
from typing import Literal
from pydantic import BaseModel, Field

from app.domain.visibility import Visibility


class PersonaCard(BaseModel):
    id: str
    name: str
    role: Literal["narrator", "npc", "companion", "antagonist"]
    public_description: str
    private_description: str | None = None
    speaking_style: str
    values: list[str] = Field(default_factory=list)
    fears: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    secrets: list[str] = Field(default_factory=list)
    forbidden_knowledge: list[str] = Field(default_factory=list)
    relationships: dict[str, str] = Field(default_factory=dict)


class SceneState(BaseModel):
    id: str
    title: str
    location: str
    current_time: str | None = None
    active_personas: list[str]
    player_visible_summary: str
    gm_private_summary: str | None = None
    open_conflicts: list[str] = Field(default_factory=list)
    active_quests: list[str] = Field(default_factory=list)
    recent_events: list[str] = Field(default_factory=list)


class MemoryEpisode(BaseModel):
    id: str
    session_id: str
    scene_id: str
    actor_id: str | None = None
    summary: str
    importance: int = Field(ge=1, le=5)
    visibility: Visibility
    tags: list[str] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    id: str
    source: str
    text: str
    score: float
    visibility: Visibility
    tags: list[str] = Field(default_factory=list)


class TurnRequest(BaseModel):
    session_id: str
    message: str
    active_persona_id: str = "narrator"


class TurnResponse(BaseModel):
    text: str
    provider: str
    model: str
    memory_written: bool
    warnings: list[str] = Field(default_factory=list)
```

The exact fields can evolve later. For MVP, this is enough.

---

## LLM Provider Interface

The app must treat local and cloud models through the same interface.

Create `app/llm/provider.py`:

```python
from abc import ABC, abstractmethod
from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class GenerationRequest(BaseModel):
    messages: list[ChatMessage]
    max_tokens: int
    temperature: float


class GenerationResult(BaseModel):
    text: str
    provider: str
    model: str


class LlmProvider(ABC):
    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        raise NotImplementedError
```

Create `app/llm/openai_compatible.py`:

```python
from openai import AsyncOpenAI

from app.llm.provider import GenerationRequest, GenerationResult, LlmProvider


class OpenAICompatibleProvider(LlmProvider):
    def __init__(self, *, base_url: str, api_key: str, model: str, provider_name: str):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.provider_name = provider_name

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[message.model_dump() for message in request.messages],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        text = response.choices[0].message.content or ""

        return GenerationResult(
            text=text,
            provider=self.provider_name,
            model=self.model,
        )
```

This works with Ollama, llama.cpp server, OpenRouter, OpenAI, and many other OpenAI-compatible APIs.

---

## Model Router

Create `app/llm/router.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class RouteDecision:
    provider: str
    reason: str
    max_tokens: int
    temperature: float


class ModelRouter:
    def __init__(self, *, cloud_mode: str, local_max_tokens: int, cloud_max_tokens: int):
        self.cloud_mode = cloud_mode
        self.local_max_tokens = local_max_tokens
        self.cloud_max_tokens = cloud_max_tokens

    def choose_route(
        self,
        *,
        intent: str,
        retrieval_confidence: float,
        failed_local_attempts: int,
        user_requested_cloud: bool = False,
    ) -> RouteDecision:
        if self.cloud_mode == "off":
            return RouteDecision("local", "cloud disabled", self.local_max_tokens, 0.75)

        if user_requested_cloud and self.cloud_mode in {"ask", "auto"}:
            return RouteDecision("cloud", "user requested cloud", self.cloud_max_tokens, 0.65)

        if self.cloud_mode == "auto" and failed_local_attempts >= 2:
            return RouteDecision("cloud", "local failed critique twice", self.cloud_max_tokens, 0.65)

        if self.cloud_mode == "auto" and intent in {"ask_lore", "scene_transition"} and retrieval_confidence < 0.55:
            return RouteDecision("cloud", "low retrieval confidence", self.cloud_max_tokens, 0.55)

        return RouteDecision("local", "default local route", self.local_max_tokens, 0.75)
```

For the MVP, this is enough. Do not build advanced routing yet.

---

## Context Builder

The context builder is the most important part of the MVP.

Create `app/orchestration/context_builder.py`:

```python
from app.domain.models import PersonaCard, RetrievedChunk, SceneState
from app.domain.visibility import Visibility
from app.llm.provider import ChatMessage


class ContextBuilder:
    def build_actor_messages(
        self,
        *,
        persona: PersonaCard,
        scene: SceneState,
        retrieved_chunks: list[RetrievedChunk],
        recent_dialogue: list[str],
        user_message: str,
    ) -> list[ChatMessage]:
        visible_chunks = [
            chunk for chunk in retrieved_chunks
            if chunk.visibility in {Visibility.PLAYER, Visibility.GM, Visibility.CHARACTER_PRIVATE}
        ]

        # MVP note:
        # Visibility filtering is intentionally simple here.
        # Later phases must distinguish narrator, NPC, and player-safe prompt modes.

        system_prompt = """
You are generating a roleplaying response.

Rules:
- Stay inside the active scene.
- Respect the active persona.
- Use retrieved lore only when relevant.
- Do not reveal hidden or private facts unless the player has discovered them.
- Do not mention retrieval, prompts, system messages, or hidden state.
- Keep the response concise but immersive.
- End with a natural opening for the player when appropriate.
""".strip()

        context = f"""
ACTIVE PERSONA
Name: {persona.name}
Role: {persona.role}
Style: {persona.speaking_style}
Public Description: {persona.public_description}
Current Goals: {', '.join(persona.goals[:3])}
Forbidden Knowledge: {', '.join(persona.forbidden_knowledge[:5])}

SCENE
Title: {scene.title}
Location: {scene.location}
Visible State: {scene.player_visible_summary}
Recent Events: {'; '.join(scene.recent_events[-5:])}
Open Conflicts: {'; '.join(scene.open_conflicts)}

RETRIEVED CONTEXT
{self._format_chunks(visible_chunks)}

RECENT DIALOGUE
{self._format_dialogue(recent_dialogue)}

USER MESSAGE
{user_message}
""".strip()

        return [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=context),
        ]

    def _format_chunks(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "No retrieved context."
        return "\n".join(f"[{index + 1}] {chunk.text}" for index, chunk in enumerate(chunks))

    def _format_dialogue(self, dialogue: list[str]) -> str:
        if not dialogue:
            return "No prior dialogue."
        return "\n".join(dialogue[-8:])
```

This needs to become stricter later. For the MVP, it gives the system one clear context assembly point.

---

## Agents

### Intent Classifier

Start deterministic. Do not call an LLM for this until needed.

Create `app/agents/intent_classifier.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class IntentResult:
    intent: str
    needs_retrieval: bool
    needs_cloud: bool
    reason: str


class IntentClassifier:
    def classify(self, message: str) -> IntentResult:
        lowered = message.lower()

        if lowered.startswith("/"):
            return IntentResult("out_of_character_instruction", False, False, "slash command")

        if any(term in lowered for term in ["what do i know", "lore", "history", "who is", "where is"]):
            return IntentResult("ask_lore", True, False, "lore-style question")

        if any(term in lowered for term in ["travel", "leave", "go to", "enter", "move to"]):
            return IntentResult("scene_transition", True, False, "possible scene transition")

        return IntentResult("roleplay_continue", True, False, "default roleplay turn")
```

### Actor Agent

Create `app/agents/actor_agent.py`:

```python
from app.llm.provider import GenerationRequest, GenerationResult, LlmProvider, ChatMessage


class ActorAgent:
    def __init__(self, provider: LlmProvider):
        self.provider = provider

    async def generate(
        self,
        *,
        messages: list[ChatMessage],
        max_tokens: int,
        temperature: float,
    ) -> GenerationResult:
        return await self.provider.generate(
            GenerationRequest(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        )
```

### Critic Agent

For the first pass, use a simple deterministic critic. Add LLM critique later.

Create `app/agents/critic_agent.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class CriticResult:
    accepted: bool
    issues: list[str]
    repair_instruction: str | None = None


class CriticAgent:
    def check(self, response: str) -> CriticResult:
        issues: list[str] = []

        forbidden_phrases = [
            "as an ai",
            "language model",
            "system prompt",
            "retrieved context",
        ]

        lowered = response.lower()
        for phrase in forbidden_phrases:
            if phrase in lowered:
                issues.append(f"Response contains forbidden assistant/meta phrase: {phrase}")

        if len(response.strip()) < 20:
            issues.append("Response is too short to be useful.")

        return CriticResult(
            accepted=not issues,
            issues=issues,
            repair_instruction="Remove meta language and answer in immersive roleplay style." if issues else None,
        )
```

### Retrieval Agent

Start with an empty stub, then wire Qdrant in the RAG phase.

Create `app/agents/retrieval_agent.py`:

```python
from app.domain.models import RetrievedChunk, SceneState, PersonaCard


class RetrievalAgent:
    async def retrieve(
        self,
        *,
        user_message: str,
        scene: SceneState,
        persona: PersonaCard,
        top_k: int,
    ) -> list[RetrievedChunk]:
        return []
```

### Memory Curator

Start with no-op memory, then add structured extraction in a later phase.

Create `app/agents/memory_curator.py`:

```python
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MemoryProposal:
    write_memory: bool
    memories: list[str] = field(default_factory=list)


class MemoryCurator:
    async def extract(self, *, user_message: str, response: str) -> MemoryProposal:
        return MemoryProposal(write_memory=False)
```

---

## Turn Orchestrator

Create `app/orchestration/turn_orchestrator.py`:

```python
from app.agents.critic_agent import CriticAgent
from app.agents.intent_classifier import IntentClassifier
from app.agents.memory_curator import MemoryCurator
from app.agents.retrieval_agent import RetrievalAgent
from app.domain.models import PersonaCard, SceneState, TurnRequest, TurnResponse
from app.llm.openai_compatible import OpenAICompatibleProvider
from app.llm.router import ModelRouter
from app.orchestration.context_builder import ContextBuilder


class TurnOrchestrator:
    def __init__(
        self,
        *,
        local_provider: OpenAICompatibleProvider,
        cloud_provider: OpenAICompatibleProvider | None,
        router: ModelRouter,
        intent_classifier: IntentClassifier,
        retrieval_agent: RetrievalAgent,
        context_builder: ContextBuilder,
        critic: CriticAgent,
        memory_curator: MemoryCurator,
    ):
        self.local_provider = local_provider
        self.cloud_provider = cloud_provider
        self.router = router
        self.intent_classifier = intent_classifier
        self.retrieval_agent = retrieval_agent
        self.context_builder = context_builder
        self.critic = critic
        self.memory_curator = memory_curator

    async def handle_turn(
        self,
        request: TurnRequest,
        *,
        scene: SceneState,
        persona: PersonaCard,
        recent_dialogue: list[str],
    ) -> TurnResponse:
        intent = self.intent_classifier.classify(request.message)

        retrieved_chunks = []
        if intent.needs_retrieval:
            retrieved_chunks = await self.retrieval_agent.retrieve(
                user_message=request.message,
                scene=scene,
                persona=persona,
                top_k=6,
            )

        route = self.router.choose_route(
            intent=intent.intent,
            retrieval_confidence=1.0 if retrieved_chunks else 0.0,
            failed_local_attempts=0,
        )

        provider = self.cloud_provider if route.provider == "cloud" else self.local_provider
        if provider is None:
            provider = self.local_provider

        messages = self.context_builder.build_actor_messages(
            persona=persona,
            scene=scene,
            retrieved_chunks=retrieved_chunks,
            recent_dialogue=recent_dialogue,
            user_message=request.message,
        )

        generation = await provider.generate(
            request={
                "messages": messages,
                "max_tokens": route.max_tokens,
                "temperature": route.temperature,
            }
        )
```

The snippet above is intentionally incomplete because it exposes an important implementation issue: `provider.generate` expects a `GenerationRequest`, not a dictionary. A coding agent must not blindly paste this broken version.

Use the corrected version:

```python
from app.llm.provider import GenerationRequest

# inside handle_turn

generation = await provider.generate(
    GenerationRequest(
        messages=messages,
        max_tokens=route.max_tokens,
        temperature=route.temperature,
    )
)

critic_result = self.critic.check(generation.text)

warnings = list(critic_result.issues)

memory_proposal = await self.memory_curator.extract(
    user_message=request.message,
    response=generation.text,
)

return TurnResponse(
    text=generation.text,
    provider=generation.provider,
    model=generation.model,
    memory_written=memory_proposal.write_memory,
    warnings=warnings,
)
```

A real file should contain only the corrected implementation.

---

## CLI First

Create a CLI before FastAPI. It makes debugging much easier.

Create `app/main.py`:

```python
import asyncio
import typer
from rich.console import Console

from app.agents.critic_agent import CriticAgent
from app.agents.intent_classifier import IntentClassifier
from app.agents.memory_curator import MemoryCurator
from app.agents.retrieval_agent import RetrievalAgent
from app.config import get_settings
from app.domain.models import PersonaCard, SceneState, TurnRequest
from app.llm.openai_compatible import OpenAICompatibleProvider
from app.llm.router import ModelRouter
from app.orchestration.context_builder import ContextBuilder
from app.orchestration.turn_orchestrator import TurnOrchestrator

cli = typer.Typer()
console = Console()


@cli.command()
def chat() -> None:
    asyncio.run(_chat())


async def _chat() -> None:
    settings = get_settings()

    local_provider = OpenAICompatibleProvider(
        base_url=settings.local_llm_base_url,
        api_key=settings.local_llm_api_key,
        model=settings.local_llm_model,
        provider_name="local",
    )

    cloud_provider = None
    if settings.cloud_mode != "off" and settings.cloud_llm_api_key != "replace_me":
        cloud_provider = OpenAICompatibleProvider(
            base_url=settings.cloud_llm_base_url,
            api_key=settings.cloud_llm_api_key,
            model=settings.cloud_llm_model,
            provider_name="cloud",
        )

    orchestrator = TurnOrchestrator(
        local_provider=local_provider,
        cloud_provider=cloud_provider,
        router=ModelRouter(
            cloud_mode=settings.cloud_mode,
            local_max_tokens=settings.local_llm_max_tokens,
            cloud_max_tokens=settings.cloud_llm_max_tokens,
        ),
        intent_classifier=IntentClassifier(),
        retrieval_agent=RetrievalAgent(),
        context_builder=ContextBuilder(),
        critic=CriticAgent(),
        memory_curator=MemoryCurator(),
    )

    scene = SceneState(
        id="starting_room",
        title="The Candlelit Room",
        location="A small stone room beneath an old tavern",
        active_personas=["narrator"],
        player_visible_summary="You stand in a candlelit room. A locked wooden door waits to the north.",
        recent_events=[],
    )

    persona = PersonaCard(
        id="narrator",
        name="Narrator",
        role="narrator",
        public_description="A concise fantasy narrator.",
        speaking_style="immersive, clear, grounded, not flowery",
        goals=["present the world", "react to the player", "keep the scene moving"],
    )

    recent_dialogue: list[str] = []

    console.print("[bold]RoleRAG POC[/bold] — type 'exit' to quit")

    while True:
        user_message = console.input("\n[bold cyan]You:[/bold cyan] ")
        if user_message.strip().lower() in {"exit", "quit"}:
            break

        response = await orchestrator.handle_turn(
            TurnRequest(session_id="dev", message=user_message),
            scene=scene,
            persona=persona,
            recent_dialogue=recent_dialogue,
        )

        console.print(f"\n[bold green]Narrator:[/bold green] {response.text}")

        recent_dialogue.append(f"User: {user_message}")
        recent_dialogue.append(f"Narrator: {response.text}")


if __name__ == "__main__":
    cli()
```

This gives the project an immediate runnable target.

---

## Docker Compose

Create `docker-compose.yml`:

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - ./data/qdrant:/qdrant/storage
```

Do not containerize the Python app yet. Do not containerize Ollama yet. Keep early debugging simple.

---

## README MVP Content

The first README should be practical:

```md
# RoleRAG POC

Personal roleplaying RAG proof of concept using Python, a local 8B model, and optional cloud fallback.

## Goals

- Local-first roleplay engine
- Structured personas and scenes
- RAG-based context assembly
- Durable memory
- Optional cloud fallback

## Quick Start

1. Install dependencies.
2. Start Ollama or llama.cpp server.
3. Copy `.env.example` to `.env`.
4. Run the CLI.

```bash
python -m app.main chat
```

## Current MVP Status

The first milestone is a local-model chat loop with structured scene and persona context.
```

---

## First Tests

Add tests immediately. Do not wait until the architecture is large.

### Test: Intent Classification

```python
from app.agents.intent_classifier import IntentClassifier


def test_default_roleplay_intent():
    result = IntentClassifier().classify("I open the wooden door.")

    assert result.intent == "roleplay_continue"
    assert result.needs_retrieval is True


def test_lore_question_intent():
    result = IntentClassifier().classify("What do I know about this city?")

    assert result.intent == "ask_lore"
    assert result.needs_retrieval is True
```

### Test: Critic Rejects Meta Language

```python
from app.agents.critic_agent import CriticAgent


def test_critic_rejects_ai_language():
    result = CriticAgent().check("As an AI language model, I cannot continue.")

    assert result.accepted is False
    assert result.issues
```

### Test: Context Builder Includes Scene

```python
from app.domain.models import PersonaCard, SceneState
from app.orchestration.context_builder import ContextBuilder


def test_context_builder_includes_scene_title():
    persona = PersonaCard(
        id="narrator",
        name="Narrator",
        role="narrator",
        public_description="A narrator.",
        speaking_style="clear",
    )
    scene = SceneState(
        id="room",
        title="Room",
        location="Stone room",
        active_personas=["narrator"],
        player_visible_summary="A locked door waits north.",
    )

    messages = ContextBuilder().build_actor_messages(
        persona=persona,
        scene=scene,
        retrieved_chunks=[],
        recent_dialogue=[],
        user_message="I inspect the door.",
    )

    joined = "\n".join(message.content for message in messages)

    assert "Room" in joined
    assert "A locked door waits north" in joined
```

---

## Important Implementation Warning

The MVP must not add RAG, memory, critic, FastAPI, world graph, and cloud fallback all in one coding pass.

The correct sequence is:

```text
1. Project skeleton
2. Config
3. Domain models
4. Local/cloud provider interface
5. CLI chat loop
6. Context builder
7. Basic deterministic intent classifier
8. Basic deterministic critic
9. Tests
10. Then RAG
11. Then memory
12. Then FastAPI
```

The first working system can have empty retrieval and no-op memory. That is acceptable. The important part is that the pipeline shape is correct.

---

## Coding Agent Instructions

Use this exact instruction when asking an LLM coding agent to implement the MVP skeleton:

```text
Implement the first MVP skeleton for RoleRAG_POC.

This is a new Python repository.
Do not reference or copy any previous Rust project.
Do not implement the full RAG system yet.

Create:
- pyproject.toml
- .env.example
- .gitignore
- README.md
- docker-compose.yml with Qdrant only
- app/config.py
- app/domain/models.py
- app/domain/visibility.py
- app/llm/provider.py
- app/llm/openai_compatible.py
- app/llm/router.py
- app/agents/intent_classifier.py
- app/agents/actor_agent.py
- app/agents/critic_agent.py
- app/agents/retrieval_agent.py with empty retrieval
- app/agents/memory_curator.py with no-op memory
- app/orchestration/context_builder.py
- app/orchestration/turn_orchestrator.py
- app/main.py with a Typer CLI chat command
- basic unit tests for intent classifier, critic, and context builder

Hard rules:
- Keep local and cloud providers behind the same interface.
- Local model is default.
- Cloud is optional.
- LLMs must not own authoritative state.
- Retrieval may be a stub in this phase.
- Memory may be a no-op in this phase.
- The CLI must run against an OpenAI-compatible local endpoint such as Ollama or llama.cpp.
- Do not add FastAPI routes yet.
- Do not add LangChain or LangGraph yet.
```

---

## MVP Skeleton Acceptance Criteria

The skeleton phase is complete when:

- `python -m app.main chat` starts a CLI session.
- the CLI sends a structured scene/persona prompt to the local model.
- the provider interface can point to Ollama or llama.cpp.
- the cloud provider can be configured but is not required.
- the actor response passes through a critic check.
- recent dialogue is kept in memory for the running process.
- retrieval is present as a stub.
- memory curation is present as a stub.
- unit tests pass.

---

## What Comes Next

The next document should define the agent workflows in more detail:

```text
docs/04_agent_workflows.md
```

That document should specify each agent's inputs, outputs, schemas, retry behavior, and failure behavior.
