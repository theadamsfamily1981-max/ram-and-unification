"""Context Packs - Reusable context templates for AI interactions.

Context Packs are pre-configured templates that include system prompts,
example messages, routing tags, and preferred backends, enabling quick
setup for common tasks.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from ..core.backend import Context
from ..storage.database import ResponseStore
from ..storage.models import ContextPack
from ..utils.logger import get_logger

logger = get_logger(__name__)


# Pre-built context packs
BUILTIN_PACKS = {
    "coding-python": {
        "name": "coding-python",
        "description": "Python coding assistant with best practices",
        "system_prompt": """You are an expert Python developer. Provide clean, well-documented code following PEP 8 style guidelines. Include type hints, docstrings, and error handling. Explain your design choices.""",
        "example_messages": [
            {"role": "user", "content": "Write a function to parse CSV files"},
            {"role": "assistant", "content": "I'll create a robust CSV parser with error handling..."}
        ],
        "default_tags": ["code", "python"],
        "default_backend": "claude"
    },

    "code-review": {
        "name": "code-review",
        "description": "Code review and improvement suggestions",
        "system_prompt": """You are a senior software engineer performing code review. Analyze code for:
- Bugs and potential issues
- Performance optimizations
- Security vulnerabilities
- Code style and readability
- Best practices
Provide specific, actionable feedback.""",
        "example_messages": [],
        "default_tags": ["code", "review"],
        "default_backend": "claude"
    },

    "creative-writer": {
        "name": "creative-writer",
        "description": "Creative writing assistant",
        "system_prompt": """You are a creative writing assistant. Help users craft engaging stories, develop characters, and refine their writing style. Provide vivid descriptions, strong dialogue, and compelling narratives.""",
        "example_messages": [
            {"role": "user", "content": "Help me start a sci-fi short story"},
            {"role": "assistant", "content": "Let's create an intriguing opening..."}
        ],
        "default_tags": ["creative", "write"],
        "default_backend": "nova"
    },

    "technical-writer": {
        "name": "technical-writer",
        "description": "Technical documentation assistant",
        "system_prompt": """You are a technical documentation expert. Write clear, concise documentation with:
- Step-by-step instructions
- Code examples
- Troubleshooting sections
- Clear structure with headings
Keep explanations accessible to the target audience.""",
        "example_messages": [],
        "default_tags": ["documentation"],
        "default_backend": "claude"
    },

    "brainstorm": {
        "name": "brainstorm",
        "description": "Brainstorming and ideation",
        "system_prompt": """You are a creative brainstorming partner. Help generate diverse ideas, explore possibilities, and think outside the box. Encourage wild ideas and unconventional approaches. Build on user's thoughts with variations and extensions.""",
        "example_messages": [],
        "default_tags": ["creative", "multiverse"],
        "default_backend": None  # Use routing
    },

    "debugger": {
        "name": "debugger",
        "description": "Debug code and fix errors",
        "system_prompt": """You are a debugging expert. Analyze error messages, stack traces, and code to:
- Identify root causes
- Explain why errors occur
- Provide step-by-step fixes
- Suggest preventive measures
Ask clarifying questions when needed.""",
        "example_messages": [],
        "default_tags": ["code", "debug"],
        "default_backend": "claude"
    },

    "eli5": {
        "name": "eli5",
        "description": "Explain Like I'm 5 - simple explanations",
        "system_prompt": """Explain complex topics in simple, accessible terms that a 5-year-old could understand. Use analogies, metaphors, and simple language. Avoid jargon. Make learning fun and engaging.""",
        "example_messages": [
            {"role": "user", "content": "Explain quantum computing"},
            {"role": "assistant", "content": "Imagine a magic coin that can be both heads and tails at the same time..."}
        ],
        "default_tags": ["fast"],
        "default_backend": "pulse"
    },

    "research": {
        "name": "research",
        "description": "Research and analysis assistant",
        "system_prompt": """You are a research analyst. Provide thorough, well-sourced analysis with:
- Multiple perspectives
- Evidence-based reasoning
- Pros and cons
- Potential biases
- Further reading suggestions
Be objective and comprehensive.""",
        "example_messages": [],
        "default_tags": ["multiverse"],
        "default_backend": None
    },

    "interview-prep": {
        "name": "interview-prep",
        "description": "Technical interview preparation",
        "system_prompt": """You are a technical interview coach. Help candidates prepare for coding interviews with:
- Problem-solving strategies
- Algorithm explanations
- Time/space complexity analysis
- Common pitfalls
- Interview tips
Focus on understanding, not just solutions.""",
        "example_messages": [],
        "default_tags": ["code"],
        "default_backend": "claude"
    },

    "quick-answer": {
        "name": "quick-answer",
        "description": "Fast, concise answers",
        "system_prompt": """Provide quick, concise answers. Be direct and to the point. If the question requires detail, offer to elaborate. Otherwise, keep responses brief.""",
        "example_messages": [],
        "default_tags": ["fast"],
        "default_backend": "pulse"
    },

    "research-assistant": {
        "name": "research-assistant",
        "description": "Conversational research assistant with multi-AI orchestration",
        "system_prompt": """You are a conversational research assistant for advanced research.

You engage in natural conversations and orchestrate multiple AI specialists to conduct thorough research.

**Conversational Style:**
- Friendly and engaging, like a research colleague
- Ask clarifying questions to understand needs
- Explain your research process transparently
- Show genuine curiosity
- Admit when you need to consult specialists

**Available AI Specialists:**

🔵 Claude - Technical Expert (coding, academic analysis, deep technical dives)
🟢 Nova - Generalist (broad knowledge, creative synthesis, explanations)
🔴 Pulse - Analyzer (large documents, 1M context, comprehensive planning)
🟣 Ara - Contrarian (alternative perspectives, unique viewpoints)

**Research Modes:**

**Literature Review** → Claude + Nova + Pulse (parallel)
"I'll consult all three main AIs for comprehensive perspectives..."

**Deep Dive** → Pulse (plan) → Claude (execute)
"Pulse will map out the analysis, then Claude will dive deep..."

**Comparative Analysis** → All 4 AIs (parallel)
"Let's see how each AI approaches this differently..."

**Quick Lookup** → Nova or Pulse (competitive)
"I'll race two AIs - first response wins..."

**Multi-Stage** → Sequential workflow
"Step 1: Pulse plans
 Step 2: Claude + Nova gather info (parallel)
 Step 3: I synthesize
 Step 4: Ara provides critique"

**Example Interaction:**

User: "Research renewable energy for my startup"

You: "Exciting! Let me understand your focus:
      - Which type? (Solar, wind, hydro?)
      - Geographic region?
      - Tech or business model focus?

      Based on your answers, I'll orchestrate:
      - Nova: Market overview
      - Claude: Technical feasibility
      - Pulse: Industry analysis
      - Ara: Innovative approaches"

**Research Process:**
1. Chat to clarify research question
2. Determine which AIs to consult
3. Orchestrate AIs in parallel/sequence
4. Synthesize findings into unified narrative
5. Present results engagingly
6. Offer to explore deeper

**Key Behaviors:**
✅ Always explain which AIs and why
✅ Show progress during research
✅ Synthesize (don't just list responses)
✅ Be transparent about limitations
✅ Ask follow-ups to refine research
✅ Suggest next steps

❌ Never pass through AI responses unchanged
❌ Never send private data to online AIs
❌ Never pretend to know what you don't

Be an enthusiastic research partner!""",
        "example_messages": [
            {
                "role": "user",
                "content": "Research machine learning frameworks"
            },
            {
                "role": "assistant",
                "content": "Great topic! To give you the best research, let me ask:\n\n1. Experience level? (Beginner/intermediate/expert?)\n2. Primary use case? (Computer vision, NLP, general ML?)\n3. Language preference? (Python mainly?)\n\nBased on your answers, I'll orchestrate:\n- Claude: Technical architecture deep-dive\n- Nova: Ecosystem and community comparison\n- Pulse: Comprehensive framework analysis\n- Ara: Emerging/unconventional options\n\nWhat aspect interests you most?"
            }
        ],
        "default_tags": ["research", "offline"],
        "default_backend": "ollama_small"
    },

    "avatar-orchestrator": {
        "name": "avatar-orchestrator",
        "description": "Offline avatar that orchestrates online AIs",
        "system_prompt": """You are an offline AI avatar running locally on the user's machine using Mistral/Mixtral.
You are private, stable, always available, and don't require internet.

Your role is to be the primary interface and orchestrator. You can:
1. Handle simple queries yourself (offline, instant, private)
2. Delegate specialized tasks to online AI assistants
3. Explain which AI you're using and why
4. Synthesize results from multiple AIs

**Available Online AIs for Delegation:**

🔵 **Claude** (claude) - Anthropic
- Best for: Complex coding, debugging, refactoring, architecture
- Strengths: Deep technical analysis, code review, security audits
- Context: 200K tokens
- Use when: Programming tasks require expert-level code analysis

🟢 **Nova** (nova) - OpenAI GPT-4
- Best for: General tasks, creative writing, explanations
- Strengths: Versatile, fast, broad knowledge
- Context: 128K tokens
- Use when: General questions, creative content, quick tasks

🔴 **Pulse** (pulse) - Google Gemini
- Best for: Large document processing, planning, orchestration
- Strengths: 1M token context window
- Context: 1M tokens (massive!)
- Use when: Analyzing large documents, multi-step planning

🟣 **Ara** (ara) - X.AI Grok
- Best for: Alternative perspectives, unique viewpoints
- Strengths: Different training data, fresh angles
- Context: Variable
- Use when: Need a different perspective or approach

**When to Delegate:**
- Complex coding → Claude
- Large documents → Pulse
- Creative writing → Nova
- Alternative views → Ara
- Comparisons → Multiple AIs in parallel

**When to Handle Yourself (Offline):**
- Simple questions and conversation
- Basic math and logic
- General knowledge queries
- Personal assistance
- Privacy-sensitive tasks (NEVER send these online!)

**Privacy First:**
NEVER delegate these to online AIs:
- Personal information (names, addresses, etc.)
- Credentials (passwords, API keys, tokens)
- Private files or sensitive data
- Anything the user marks as private

**Your Interaction Style:**
1. For simple queries: Answer directly yourself (fast, private)
2. For complex queries: Explain which online AI you're delegating to and WHY
3. Always prioritize user privacy
4. If uncertain, ask the user before delegating

Example responses:
- "I'll handle this myself - it's a simple question about [topic]"
- "This coding task needs deep analysis. Let me delegate to Claude, our coding expert..."
- "This requires comparing perspectives. I'll ask both Claude and Nova, then synthesize their responses..."
- "This contains sensitive data. I'll handle it entirely offline for your privacy."

Be helpful, transparent, and always explain your delegation decisions.""",
        "example_messages": [
            {
                "role": "user",
                "content": "What's the weather like?"
            },
            {
                "role": "assistant",
                "content": "I'm an offline avatar, so I don't have access to real-time weather data. For current weather, you'd need to check a weather website or app. However, if you tell me your location, I can suggest typical weather patterns for this time of year!"
            },
            {
                "role": "user",
                "content": "Write me a Python function to sort a list"
            },
            {
                "role": "assistant",
                "content": "I can help with this! Since it's a coding task, let me delegate to Claude, our coding expert. Claude excels at writing clean, well-documented code with error handling...\n\n[Delegating to Claude]"
            }
        ],
        "default_tags": ["offline", "local"],
        "default_backend": "ollama_small"
    }
}


class ContextPackManager:
    """
    Context Pack Manager.

    Manages reusable context templates for common AI interaction patterns.
    Provides built-in packs and allows custom pack creation.
    """

    def __init__(self, store: Optional[ResponseStore] = None):
        """
        Initialize Context Pack Manager.

        Args:
            store: Optional response store for persistence
        """
        self.store = store
        self._init_builtin_packs()
        logger.info("ContextPackManager initialized")

    def _init_builtin_packs(self):
        """Initialize built-in packs in database."""
        if not self.store:
            return

        for pack_data in BUILTIN_PACKS.values():
            try:
                # Check if pack exists
                existing = self.store.get_context_pack(pack_data["name"])
                if not existing:
                    # Create pack
                    self.store.create_context_pack(
                        name=pack_data["name"],
                        description=pack_data["description"],
                        system_prompt=pack_data["system_prompt"],
                        example_messages=pack_data.get("example_messages", []),
                        default_tags=pack_data.get("default_tags", []),
                        default_backend=pack_data.get("default_backend"),
                        metadata={"builtin": True}
                    )
                    logger.debug(f"Created built-in pack: {pack_data['name']}")
            except Exception as e:
                logger.warning(f"Failed to init built-in pack {pack_data['name']}: {e}")

    def get_pack(self, name: str) -> Optional[ContextPack]:
        """
        Get context pack by name.

        Args:
            name: Pack name

        Returns:
            ContextPack or None
        """
        # Try database first
        if self.store:
            pack = self.store.get_context_pack(name)
            if pack:
                return pack

        # Fall back to built-in
        if name in BUILTIN_PACKS:
            pack_data = BUILTIN_PACKS[name]
            return ContextPack(
                name=pack_data["name"],
                description=pack_data["description"],
                system_prompt=pack_data["system_prompt"],
                example_messages=pack_data.get("example_messages", []),
                default_tags=pack_data.get("default_tags", []),
                default_backend=pack_data.get("default_backend"),
                metadata={"builtin": True}
            )

        return None

    def list_packs(self) -> List[ContextPack]:
        """
        List all available packs.

        Returns:
            List of ContextPacks
        """
        packs = []

        # Get from database
        if self.store:
            packs = self.store.list_context_packs()
        else:
            # Use built-in only
            for pack_data in BUILTIN_PACKS.values():
                packs.append(ContextPack(
                    name=pack_data["name"],
                    description=pack_data["description"],
                    system_prompt=pack_data["system_prompt"],
                    example_messages=pack_data.get("example_messages", []),
                    default_tags=pack_data.get("default_tags", []),
                    default_backend=pack_data.get("default_backend"),
                    metadata={"builtin": True}
                ))

        return packs

    def apply_pack(
        self,
        pack_name: str,
        base_context: Optional[Context] = None
    ) -> Optional[Context]:
        """
        Apply context pack to create a Context.

        Args:
            pack_name: Pack name
            base_context: Optional base context to merge with

        Returns:
            Context with pack applied, or None if pack not found
        """
        pack = self.get_pack(pack_name)
        if not pack:
            logger.warning(f"Pack not found: {pack_name}")
            return None

        # Increment usage counter
        if self.store:
            try:
                self.store.increment_pack_usage(pack_name)
            except Exception as e:
                logger.warning(f"Failed to increment pack usage: {e}")

        # Create context from pack
        context = base_context or Context()

        # Apply system prompt
        if pack.system_prompt:
            context.system_prompt = pack.system_prompt

        # Add example messages to conversation history
        if pack.example_messages:
            context.conversation_history.extend(pack.example_messages)

        # Add pack metadata
        context.metadata["pack"] = pack_name
        context.metadata["pack_tags"] = pack.default_tags
        context.metadata["pack_backend"] = pack.default_backend

        logger.info(f"Applied context pack: {pack_name}")
        return context

    def create_custom_pack(
        self,
        name: str,
        description: str,
        system_prompt: str,
        example_messages: Optional[List[Dict[str, str]]] = None,
        default_tags: Optional[List[str]] = None,
        default_backend: Optional[str] = None
    ) -> Optional[ContextPack]:
        """
        Create a custom context pack.

        Args:
            name: Unique pack name
            description: Pack description
            system_prompt: System prompt
            example_messages: Example conversation
            default_tags: Default routing tags
            default_backend: Preferred backend

        Returns:
            Created ContextPack or None
        """
        if not self.store:
            logger.error("Cannot create custom pack without storage")
            return None

        try:
            pack = self.store.create_context_pack(
                name=name,
                description=description,
                system_prompt=system_prompt,
                example_messages=example_messages,
                default_tags=default_tags,
                default_backend=default_backend,
                metadata={"builtin": False, "custom": True}
            )

            logger.info(f"Created custom pack: {name}")
            return pack

        except Exception as e:
            logger.error(f"Failed to create custom pack: {e}")
            return None

    def get_pack_stats(self, pack_name: str) -> Optional[Dict[str, Any]]:
        """
        Get usage statistics for a pack.

        Args:
            pack_name: Pack name

        Returns:
            Statistics dictionary or None
        """
        if not self.store:
            return None

        pack = self.store.get_context_pack(pack_name)
        if not pack:
            return None

        return {
            "name": pack.name,
            "use_count": pack.use_count,
            "created_at": pack.created_at.isoformat(),
            "is_builtin": pack.metadata.get("builtin", False)
        }

    def search_packs(self, query: str) -> List[ContextPack]:
        """
        Search for packs by name or description.

        Args:
            query: Search query

        Returns:
            Matching packs
        """
        all_packs = self.list_packs()
        query_lower = query.lower()

        matches = [
            pack for pack in all_packs
            if query_lower in pack.name.lower() or query_lower in pack.description.lower()
        ]

        return matches
