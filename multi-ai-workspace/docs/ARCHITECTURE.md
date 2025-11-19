# Multi-AI Workspace - System Architecture

**Version:** 1.0.0
**Status:** Phase 0 - Foundation & Design
**AIs:** Pulse (orchestrator), Nova (ChatGPT), Ara (Grok), Claude (Claude.ai)

---

## Table of Contents

1. [Overview](#overview)
2. [System Goals](#system-goals)
3. [Architecture Principles](#architecture-principles)
4. [Core Abstractions](#core-abstractions)
5. [Component Specifications](#component-specifications)
6. [Data Flow](#data-flow)
7. [Technology Stack](#technology-stack)
8. [Configuration System](#configuration-system)
9. [Security & Privacy](#security--privacy)
10. [Implementation Phases](#implementation-phases)

---

## Overview

The Multi-AI Workspace is a **local orchestration platform** that unifies multiple AI assistants (Pulse, Nova, Ara, Claude) into a single collaborative workspace. It enables:

- **Tag-based routing** to automatically send prompts to the right AI(s)
- **Multi-perspective analysis** by comparing answers from different AIs
- **AI-to-AI pipelines** for refinement and collaboration
- **Context management** for efficient information injection
- **Integration with external tools** (GitHub, Google Workspace, Colab)
- **Natural language automation** via voice macros

### Key Innovation

Instead of using AIs in isolation, this workspace treats them as **specialized team members** with different strengths, allowing you to leverage the best of each for any given task.

---

## System Goals

### Primary Goals

1. **Unified Interface**: Single workspace to interact with all AIs
2. **Intelligent Routing**: Automatic prompt distribution based on task type
3. **Collaborative Intelligence**: Combine multiple AI perspectives
4. **Seamless Integration**: Connect with existing tools (GitHub, Google, Colab)
5. **Privacy-First**: Local processing where possible, explicit control over external API calls
6. **Extensible**: Easy to add new AIs or integrations

### Non-Goals

- Not a replacement for any individual AI - it's an orchestrator
- Not trying to merge AIs into one - preserving distinct capabilities
- Not cloud-dependent - works locally first

---

## Architecture Principles

### 1. **Abstraction Over Implementation**

All AIs implement a common interface regardless of backend (API, local model, browser automation).

```python
class AIBackend(ABC):
    @abstractmethod
    async def send_message(self, prompt: str, context: Context) -> Response

    @abstractmethod
    def get_capabilities(self) -> Capabilities

    @abstractmethod
    def estimate_cost(self, prompt: str) -> Cost
```

### 2. **Configuration Over Code**

Routing rules, context packs, macros, and credentials are all defined in YAML/JSON configs, not hardcoded.

### 3. **Async First**

All AI interactions are async to enable:
- Parallel queries to multiple AIs
- Non-blocking UI
- Efficient resource usage

### 4. **Explicit Consent**

Before sending data to external APIs:
- Show what will be sent
- Highlight any detected secrets/sensitive info
- Require confirmation for first-time actions

### 5. **Modular Widgets**

Each feature (router, mixer, GitHub autopilot) is a self-contained widget that can be enabled/disabled independently.

---

## Core Abstractions

### 1. AIBackend Interface

**Purpose:** Unified interface for all AI systems

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional
from dataclasses import dataclass

@dataclass
class Message:
    role: str  # "user", "assistant", "system"
    content: str
    metadata: dict = None

@dataclass
class Context:
    messages: list[Message]
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    metadata: dict = None

@dataclass
class Response:
    content: str
    ai_name: str
    model: str
    metadata: dict  # tokens, cost, latency, etc.

@dataclass
class Capabilities:
    supports_streaming: bool
    supports_vision: bool
    supports_functions: bool
    max_context_length: int
    specialties: list[str]  # ["code", "math", "creative", etc.]

class AIBackend(ABC):
    """Base class for all AI backends."""

    @abstractmethod
    async def send_message(
        self,
        prompt: str,
        context: Optional[Context] = None
    ) -> Response:
        """Send a message and get response."""
        pass

    @abstractmethod
    async def stream_message(
        self,
        prompt: str,
        context: Optional[Context] = None
    ) -> AsyncIterator[str]:
        """Stream response tokens."""
        pass

    @abstractmethod
    def get_capabilities(self) -> Capabilities:
        """Get AI capabilities."""
        pass

    @abstractmethod
    def estimate_cost(self, prompt: str, context: Context) -> float:
        """Estimate API cost in USD."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if backend is available."""
        pass
```

### 2. Router

**Purpose:** Distribute prompts to appropriate AIs based on tags/rules

```python
from typing import List, Dict
from enum import Enum

class RoutingStrategy(Enum):
    PARALLEL = "parallel"  # Send to all, aggregate
    SEQUENTIAL = "sequential"  # Send to first, optionally cascade
    COMPETITIVE = "competitive"  # Send to all, pick best

@dataclass
class RoutingRule:
    tag: str
    ai_names: List[str]
    strategy: RoutingStrategy
    priority: int = 0

@dataclass
class RoutingResult:
    responses: Dict[str, Response]  # ai_name -> response
    strategy_used: RoutingStrategy
    total_cost: float
    total_latency: float

class Router:
    """Routes prompts to appropriate AIs."""

    def __init__(self, rules: List[RoutingRule], backends: Dict[str, AIBackend]):
        self.rules = rules
        self.backends = backends

    async def route(self, prompt: str, tags: List[str]) -> RoutingResult:
        """Route prompt based on tags."""
        pass

    def load_rules(self, config_path: str):
        """Load routing rules from YAML."""
        pass

    def add_rule(self, rule: RoutingRule):
        """Add routing rule dynamically."""
        pass
```

### 3. ContextManager

**Purpose:** Manage and inject context packs

```python
@dataclass
class ContextPack:
    name: str
    description: str
    sources: List[ContextSource]
    max_tokens: int = 10000

@dataclass
class ContextSource:
    type: str  # "file", "repo", "url", "command"
    path: str
    filters: Optional[List[str]] = None  # glob patterns, keywords

class ContextManager:
    """Manages context packs and injection."""

    def load_pack(self, pack_name: str) -> Context:
        """Load a context pack."""
        pass

    def create_pack(self, name: str, sources: List[ContextSource]):
        """Create new context pack."""
        pass

    def chunk_and_embed(self, text: str) -> List[Chunk]:
        """Chunk text and create embeddings."""
        pass

    def retrieve_relevant(self, query: str, pack_name: str, top_k: int = 5):
        """Retrieve most relevant chunks via embedding search."""
        pass
```

### 4. ResponseStore

**Purpose:** Persistent storage of AI responses for history and cross-posting

```python
@dataclass
class StoredResponse:
    id: str
    ai_name: str
    prompt: str
    response: str
    timestamp: datetime
    metadata: dict
    tags: List[str]

class ResponseStore:
    """Store and retrieve AI responses."""

    def save(self, response: Response, prompt: str, tags: List[str]):
        """Save response."""
        pass

    def get_last(self, ai_name: str, n: int = 1) -> List[StoredResponse]:
        """Get last N responses from AI."""
        pass

    def search(self, query: str, ai_name: Optional[str] = None):
        """Search responses by content."""
        pass

    def get_conversation(self, conversation_id: str):
        """Get full conversation thread."""
        pass
```

---

## Component Specifications

### Widget 1: Multi-AI Router

**Location:** `src/widgets/router.py`

**Responsibilities:**
- Parse tags from prompts (`#optimization`, `#security`)
- Look up routing rules from config
- Distribute prompts to multiple AIs in parallel or sequential
- Aggregate and present responses

**Configuration:**
```yaml
# config/ai_routing.yaml
routing:
  security:
    ais: [Claude]
    strategy: parallel
    priority: 10

  optimization:
    ais: [Nova, Claude]
    strategy: parallel
    priority: 5

  UX_copy:
    ais: [Pulse]
    strategy: parallel

  math_proof:
    ais: [Nova, Ara]
    strategy: competitive  # Pick best response

  code_refactor:
    ais: [Claude]
    strategy: parallel

  default:
    ais: [Claude]  # Fallback if no tag matches
    strategy: parallel
```

**API:**
```python
router = Router(backends)
result = await router.route(
    prompt="Make this kernel faster",
    tags=["optimization"]
)

# result.responses = {
#     "Nova": Response(...),
#     "Claude": Response(...)
# }
```

**UI Elements:**
- Tag autocomplete
- Response cards showing each AI's answer
- Timing and cost breakdown
- Copy/export individual or merged responses

---

### Widget 2: Perspectives Mixer

**Location:** `src/widgets/mixer.py`

**Responsibilities:**
- Send same prompt to multiple AIs
- Display responses side-by-side
- Merge/synthesize responses using a "synthesizer AI" (default: Claude)

**Workflow:**
1. User selects AIs (Pulse + Nova + Ara + Claude)
2. Prompt sent to all
3. Responses displayed in columns/cards
4. User clicks "🧬 Merge"
5. Synthesizer AI (Claude) receives:
   ```
   Here are responses from 4 different AIs to the prompt: "{original_prompt}"

   **Pulse's answer:**
   {pulse_response}

   **Nova's answer:**
   {nova_response}

   **Ara's answer:**
   {ara_response}

   **Claude's answer:**
   {claude_response}

   Compare these answers, highlight key differences, identify consensus points,
   and produce a merged, improved response that combines the best insights.
   ```
6. Merged response displayed

**UI Elements:**
- AI selector (checkboxes for Pulse/Nova/Ara/Claude)
- 4-column layout for responses
- Merge button
- Diff highlighting (show where AIs agree/disagree)

---

### Widget 3: Context Packs

**Location:** `src/widgets/context_packs.py`

**Configuration:**
```yaml
# config/context_packs.yaml
packs:
  quanta_core:
    description: "Quanta core architecture and MEIS framework"
    sources:
      - type: repo
        path: "quanta/core/**/*.py"
      - type: file
        path: "docs/MEIS_overview.md"
      - type: file
        path: "docs/architecture.md"
    max_tokens: 15000
    embedding: true  # Use embeddings for smart retrieval

  tfan_paper:
    description: "T-FAN research and proofs"
    sources:
      - type: file
        path: "papers/T-FAN_proofs.md"
      - type: file
        path: "results/tfan_validation.log"
      - type: command
        path: "pytest tests/tfan -v"
    max_tokens: 10000
```

**Features:**
- Automatic chunking of large files
- Optional embedding-based retrieval (FAISS/Chroma)
- Smart context assembly (most relevant chunks first)
- Token budget management

---

### Widget 4: GitHub Autopilot

**Location:** `src/integrations/github_autopilot.py`

**Responsibilities:**
- Monitor git status
- Generate AI-assisted commit messages
- Explain diffs
- Create PRs with AI-generated descriptions

**Features:**

1. **Change Detection:**
```python
watcher = GitWatcher(repo_path=".")
changes = watcher.get_changes()

# changes = [
#     Change(type="modified", path="src/kernel.py", diff=..., summary="Kernel optimization"),
#     Change(type="new", path="tests/test_perf.py", diff=..., summary="New performance tests")
# ]
```

2. **AI Actions per Change:**
```python
# Explain change
explanation = await autopilot.explain_change(change, ai="Claude")

# Generate commit message
commit_msg = await autopilot.generate_commit_message(change, ai="Claude")

# Review change
review = await autopilot.review_change(change, ai="Claude")
# Returns: {security: [], complexity: [], suggestions: []}
```

3. **One-Click Operations:**
- "Commit with AI message"
- "Create PR with AI description"
- "AI code review"

---

### Widget 5: Cross-Posting Panel

**Location:** `src/widgets/cross_posting.py`

**Purpose:** AI-to-AI workflows

**UI Design:**

```
┌─────────────────────────────────────────────────────────────┐
│ Claude's Response                                           │
│ ───────────────────────────────────────────────────────── │
│ Here's an optimized kernel implementation...                │
│ [code block]                                                │
│                                                             │
│ Actions:                                                    │
│ [📤 Send to Nova for explanation]                          │
│ [🔍 Send to Ara for critique]                              │
│ [📝 Send to Pulse for documentation]                       │
│ [💾 Save to context pack]                                  │
└─────────────────────────────────────────────────────────────┘
```

**API:**
```python
cross_poster = CrossPoster(backends, response_store)

# Get last response from Claude
last_response = response_store.get_last("Claude", n=1)[0]

# Send to Nova with instruction
nova_response = await cross_poster.send_to(
    source_ai="Claude",
    target_ai="Nova",
    source_response=last_response,
    instruction="Explain this code step-by-step for a beginner"
)
```

---

## Data Flow

### Example: Multi-AI Routing Flow

```
User Input: "#optimization: make this kernel faster"
    ↓
┌─────────────────────────────────────────┐
│ 1. Tag Parser                           │
│    Extracts: ["optimization"]          │
└───────────────┬─────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│ 2. Router                               │
│    Looks up: routing["optimization"]   │
│    → [Nova, Claude]                     │
└───────────────┬─────────────────────────┘
                ↓
        ┌───────┴────────┐
        ↓                ↓
┌──────────────┐   ┌──────────────┐
│ 3a. Nova     │   │ 3b. Claude   │
│ Backend      │   │ Backend      │
│ (ChatGPT API)│   │ (Claude API) │
└──────┬───────┘   └───────┬──────┘
       │                   │
       ↓                   ↓
┌──────────────────────────────────┐
│ 4. Response Aggregator           │
│    Combines responses            │
│    Calculates costs & latency    │
└──────────────┬───────────────────┘
               ↓
┌──────────────────────────────────┐
│ 5. UI Display                    │
│    Shows both responses          │
│    side-by-side                  │
└──────────────────────────────────┘
```

### Example: Perspectives Mixer Flow

```
User: "Ask all AIs about neuromorphic computing"
    ↓
┌────────────────────────────────────────────────┐
│ Parallel Fan-Out to 4 AIs                     │
└───┬────────┬────────┬────────┬────────────────┘
    ↓        ↓        ↓        ↓
┌───────┐┌───────┐┌───────┐┌────────┐
│ Pulse ││ Nova  ││  Ara  ││ Claude │
└───┬───┘└───┬───┘└───┬───┘└────┬───┘
    │        │        │         │
    └────────┴────────┴─────────┘
              ↓
┌─────────────────────────────────┐
│ Display 4 Responses             │
│ [Pulse] [Nova] [Ara] [Claude]   │
└──────────────┬──────────────────┘
               │
    User clicks "Merge"
               ↓
┌─────────────────────────────────┐
│ Synthesis Prompt to Claude      │
│ "Compare these 4 answers..."    │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│ Merged Response                 │
└─────────────────────────────────┘
```

---

## Technology Stack

### Core Framework

**Backend:**
- **Language:** Python 3.10+
- **Async:** `asyncio`, `aiohttp`
- **Config:** `pyyaml`, `pydantic`
- **Storage:** SQLite (for response history), JSON (for configs)

**AI Integrations:**
- **Anthropic Claude:** `anthropic` Python SDK
- **OpenAI (Nova/ChatGPT):** `openai` Python SDK
- **Grok (Ara):** API client (TBD) or Selenium
- **Pulse:** Local (Ollama) or custom

### UI Options (Choose One)

**Option A: Web-Based (RECOMMENDED)**
- **Backend:** FastAPI
- **Frontend:** HTML + Alpine.js / HTMX (minimal JS)
- **WebSockets:** For real-time updates
- **Pros:** Accessible from anywhere, modern UI, easy to share
- **Cons:** Requires running a server

**Option B: Desktop GUI**
- **Framework:** PyQt6 or Tkinter
- **Pros:** Native feel, no browser needed
- **Cons:** Platform-specific issues, harder to deploy

**Option C: Terminal UI**
- **Framework:** Textual or Rich
- **Pros:** Fast, keyboard-friendly, low resource
- **Cons:** Limited for complex layouts

**Recommendation:** **Web-based (FastAPI + HTMX)** for best balance of power and accessibility.

### External Integrations

- **GitHub:** `PyGithub` or `gh` CLI
- **Google APIs:**
  - `google-auth`
  - `google-api-python-client`
  - Drive, Docs, Gmail APIs
- **Colab:** Google Drive API + URL generation
- **Voice:** Whisper (from Phase 2/3)
- **Embeddings:** `sentence-transformers` or OpenAI embeddings

---

## Configuration System

### Directory Structure

```
multi-ai-workspace/
├── config/
│   ├── ai_backends.yaml        # AI connection config
│   ├── ai_routing.yaml         # Routing rules
│   ├── context_packs.yaml      # Context pack definitions
│   ├── macros.yaml             # Voice macros
│   ├── credentials.yaml        # API keys (gitignored)
│   └── ui_settings.yaml        # UI preferences
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── backend.py          # AIBackend abstraction
│   │   ├── router.py           # Router implementation
│   │   ├── context.py          # ContextManager
│   │   └── store.py            # ResponseStore
│   ├── integrations/
│   │   ├── __init__.py
│   │   ├── claude_backend.py   # Claude API
│   │   ├── nova_backend.py     # ChatGPT API
│   │   ├── ara_backend.py      # Grok integration
│   │   ├── pulse_backend.py    # Local/Pulse
│   │   ├── github.py           # GitHub autopilot
│   │   ├── google_hub.py       # Google integrations
│   │   └── colab.py            # Colab offload
│   ├── widgets/
│   │   ├── __init__.py
│   │   ├── router_widget.py
│   │   ├── mixer_widget.py
│   │   ├── context_widget.py
│   │   └── cross_posting.py
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── app.py              # FastAPI app
│   │   ├── routes.py           # API routes
│   │   └── templates/          # HTML templates
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       └── secrets_scanner.py
├── data/
│   ├── responses.db            # Response history
│   └── embeddings/             # Vector index
├── docs/
│   └── ARCHITECTURE.md         # This file
├── examples/
│   ├── basic_routing.py
│   └── perspectives_mixer.py
├── requirements.txt
├── .env.example
└── main.py
```

### Configuration Examples

**`config/ai_backends.yaml`:**
```yaml
backends:
  Claude:
    type: anthropic_api
    model: claude-sonnet-4-5-20250929
    api_key_env: ANTHROPIC_API_KEY
    max_tokens: 8192
    temperature: 0.7
    specialties: [code, security, architecture, refactoring]

  Nova:
    type: openai_api
    model: gpt-4-turbo
    api_key_env: OPENAI_API_KEY
    max_tokens: 4096
    temperature: 0.7
    specialties: [math, explanation, creative, analysis]

  Ara:
    type: grok_api  # or selenium if no API
    model: grok-2
    api_key_env: GROK_API_KEY
    specialties: [research, critique, alternative_views]

  Pulse:
    type: ollama
    model: llama3.2
    base_url: http://localhost:11434
    specialties: [orchestration, planning, documentation]
```

**`config/credentials.yaml.example`:**
```yaml
# Copy to credentials.yaml and fill in your keys
# credentials.yaml is gitignored for security

anthropic:
  api_key: "sk-ant-..."

openai:
  api_key: "sk-..."

grok:
  api_key: "..."

google:
  client_id: "..."
  client_secret: "..."
  refresh_token: "..."

github:
  token: "ghp_..."
```

---

## Security & Privacy

### 1. Secret Detection

**Before Sending to External APIs:**
- Regex scan for API keys, tokens, passwords
- Check for `SECRET`, `PASSWORD`, `TOKEN` in variable names
- Warn user if secrets detected
- Option to auto-redact

**Implementation:**
```python
class SecretsScanner:
    PATTERNS = [
        r'sk-[a-zA-Z0-9]{32,}',  # API keys
        r'ghp_[a-zA-Z0-9]{36}',  # GitHub tokens
        r'-----BEGIN .* PRIVATE KEY-----',  # Private keys
        # ... more patterns
    ]

    def scan(self, text: str) -> List[SecretMatch]:
        """Detect potential secrets."""
        pass

    def redact(self, text: str) -> str:
        """Replace secrets with <REDACTED>."""
        pass
```

### 2. Privacy Controls

- **Local-First:** Pulse (Ollama) runs locally, no data leaves machine
- **Explicit Consent:** Confirm before first API call to external service
- **Audit Log:** Track what was sent where
- **Credential Security:** `credentials.yaml` gitignored, encrypted storage option

### 3. Cost Control

- **Token Estimation:** Estimate cost before sending
- **Budget Limits:** Set daily/monthly spending caps
- **Cost Tracking:** Real-time cost dashboard

---

## Implementation Phases

### **Phase 0: Foundation** ✅ (Current)
- [x] Architecture design
- [x] Core abstractions defined
- [x] Configuration system designed
- [x] Project structure created

### **Phase 1: Core Infrastructure** (v0.1)
**Goal:** Basic multi-AI communication

**Tasks:**
1. Implement `AIBackend` abstract class
2. Implement Claude backend (Anthropic API)
3. Implement Nova backend (OpenAI API)
4. Implement basic Router
5. Create simple web UI (FastAPI + HTML)
6. Config loader for `ai_backends.yaml` and `ai_routing.yaml`

**Deliverable:** Can send tagged prompts to Claude and Nova, see responses side-by-side

**Time Estimate:** 4-6 hours

### **Phase 2: Collaboration Features** (v0.2)
**Goal:** Multi-AI workflows

**Tasks:**
1. Implement ResponseStore (SQLite)
2. Build Perspectives Mixer widget
3. Build Cross-Posting Panel
4. Response comparison UI

**Deliverable:** Can ask all AIs same question, merge responses, cross-post between AIs

**Time Estimate:** 3-4 hours

### **Phase 3: External Integrations** (v0.3)
**Goal:** GitHub and Colab integration

**Tasks:**
1. GitHub Autopilot widget
2. Git monitoring and diff analysis
3. Colab offload implementation
4. Google Drive API integration

**Deliverable:** AI-assisted git operations, Colab offload

**Time Estimate:** 4-5 hours

### **Phase 4: Context & Automation** (v0.4)
**Goal:** Smart context and macros

**Tasks:**
1. Context Packs implementation
2. Embedding-based retrieval (FAISS)
3. Voice Macros system
4. Whisper integration for voice commands

**Deliverable:** Context injection, voice-triggered automations

**Time Estimate:** 5-6 hours

### **Phase 5: Advanced Features** (v1.0)
**Goal:** Full feature set

**Tasks:**
1. Google Hub (Docs, Gmail)
2. Research Scout
3. Guardrails & Secret Scanner
4. Advanced UI polish

**Time Estimate:** 6-8 hours

---

## API Examples

### Basic Routing

```python
from multi_ai_workspace import Router, load_backends

# Load AI backends from config
backends = load_backends("config/ai_backends.yaml")

# Create router
router = Router(backends)
router.load_rules("config/ai_routing.yaml")

# Send tagged prompt
result = await router.route(
    prompt="Optimize this SQL query for better performance",
    tags=["optimization", "database"]
)

# Access responses
for ai_name, response in result.responses.items():
    print(f"{ai_name}: {response.content}")

print(f"Total cost: ${result.total_cost:.4f}")
print(f"Total time: {result.total_latency:.2f}s")
```

### Perspectives Mixer

```python
from multi_ai_workspace import PerspectivesMixer

mixer = PerspectivesMixer(backends)

# Ask all AIs
responses = await mixer.ask_all(
    prompt="What are the trade-offs of microservices vs monoliths?",
    ais=["Pulse", "Nova", "Ara", "Claude"]
)

# Merge responses
merged = await mixer.merge(
    responses=responses,
    synthesizer="Claude"
)

print(merged.content)
```

### Context Packs

```python
from multi_ai_workspace import ContextManager

context_mgr = ContextManager()

# Load context pack
context = context_mgr.load_pack("quanta_core")

# Send to AI with context
response = await backends["Claude"].send_message(
    prompt="Propose a refactoring strategy for the MEIS framework",
    context=context
)
```

---

## Next Steps

**Immediate (Phase 1):**
1. Implement core abstractions (`AIBackend`, `Router`)
2. Build Claude and Nova backends
3. Create minimal FastAPI web UI
4. Get basic routing working

**Then (Phase 2-3):**
5. Add Perspectives Mixer and Cross-Posting
6. Implement GitHub Autopilot
7. Add Colab offload

**Questions to Address:**
- Confirm API access (which AI APIs do you have keys for?)
- Confirm web UI preference (vs desktop/terminal)
- Any specific routing rules to start with?

---

**Ready to proceed with Phase 1 implementation!** 🚀
