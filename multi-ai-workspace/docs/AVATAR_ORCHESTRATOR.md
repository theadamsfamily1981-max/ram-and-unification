# Avatar Orchestrator - Hybrid Offline/Online AI Architecture

**Version:** 1.1.0
**Date:** 2025-01-18

## Overview

The **Avatar Orchestrator** is a hybrid AI system that combines the stability and privacy of offline AI with the specialized capabilities of online AI services.

### Architecture

```
┌─────────────────────────────────────────────┐
│         OFFLINE AVATAR (Primary)            │
│      Mistral 7B / Mixtral 8x7B MoE          │
│                                             │
│  • Runs 100% locally (private)              │
│  • Always available (no internet needed)    │
│  • Handles simple queries instantly         │
│  • Orchestrates online AIs when needed      │
└─────────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │   Delegates to Online   │
        │   AIs for specialized   │
        │        tasks            │
        └────────────┬────────────┘
                     │
    ┌────────────────┼────────────────┬──────────┐
    ▼                ▼                ▼          ▼
┌────────┐      ┌────────┐      ┌────────┐  ┌────────┐
│ Claude │      │  Nova  │      │ Pulse  │  │  Ara   │
│Coding  │      │General │      │Docs    │  │Unique  │
│Expert  │      │Purpose │      │1M ctx  │  │Views   │
└────────┘      └────────┘      └────────┘  └────────┘
```

## Key Concept: Offline Avatar as Primary Interface

**The avatar is your stable, private, offline AI that:**
1. **Handles simple queries itself** - Fast, private, no internet needed
2. **Delegates to online AIs** - For complex/specialized tasks
3. **Explains delegation** - Tells you which AI it's using and why
4. **Protects privacy** - Never sends sensitive data online

## Why This Architecture?

### Problem: Online-Only AI
- ❌ Requires internet (not always available)
- ❌ Costs money for every request
- ❌ Privacy concerns (data sent to cloud)
- ❌ Network latency
- ❌ Service outages

### Problem: Offline-Only AI
- ❌ Limited capabilities vs GPT-4/Claude
- ❌ Can't access specialized models
- ❌ Smaller context windows
- ❌ No real-time information

### Solution: Hybrid Avatar Orchestrator
- ✅ **Offline by default** - Privacy, speed, always available
- ✅ **Online when needed** - Access to specialized AIs
- ✅ **Transparent** - Avatar explains its decisions
- ✅ **Cost-effective** - Only use expensive APIs when necessary
- ✅ **Best of both worlds**

## How It Works

### User Flow Example

**Simple Query (Handled Offline):**
```
User: "What's 25 * 37?"

Avatar: "925. I calculated this offline - simple math doesn't need
         online AI assistance."
```

**Complex Coding Query (Delegated to Claude):**
```
User: "Review this Python code for security vulnerabilities"

Avatar: "This requires expert-level code analysis. I'm delegating to
         Claude, our coding specialist, because:

         - Claude excels at security audits
         - It has 200K token context for large codebases
         - Best at identifying OWASP vulnerabilities

         [Delegating to Claude...]

         [Claude's analysis follows...]"
```

**Multi-AI Comparison (Parallel Delegation):**
```
User: "What's the best web framework for startups?"

Avatar: "This is subjective and benefits from multiple perspectives.
         I'm asking Claude, Nova, and Pulse in parallel, then I'll
         synthesize their recommendations...

         [Delegating to 3 AIs in parallel...]

         Here's what each AI recommends:
         - Claude: Next.js (best for full-stack TypeScript)
         - Nova: FastAPI (Python, rapid prototyping)
         - Pulse: Django (batteries-included, scalable)

         My synthesis: ..."
```

## Delegation Rules

The avatar decides when to delegate based on:

### 1. Task Complexity

| Complexity | Handled By | Example |
|------------|-----------|---------|
| Simple | Avatar (offline) | Basic math, general questions |
| Medium | Avatar or online | Depends on topic |
| Complex | Online AI | Code review, large document analysis |

### 2. Task Type

| Task Type | Delegated To | Reason |
|-----------|-------------|--------|
| Coding/debugging | Claude | Expert code analysis |
| Large documents (>10K words) | Pulse | 1M token context |
| Creative writing | Nova | Excellent at creative content |
| Alternative perspective | Ara | Unique training data |
| Privacy-sensitive | Avatar (offline) | Never sent online |

### 3. Privacy Level

| Data Type | Handling |
|-----------|----------|
| Public info | Can delegate to online AIs |
| Personal info | **OFFLINE ONLY** |
| Credentials (passwords, keys) | **OFFLINE ONLY** |
| Sensitive data | **OFFLINE ONLY** |

## Configuration

### Avatar Configuration File

**File:** `config/avatar.yaml`

```yaml
avatar:
  name: "Avatar"
  backend: ollama_small  # Offline Mistral 7B

  # Delegation rules
  delegation_rules:
    - trigger:
        keywords: [code, coding, debug]
        complexity: medium_to_high
      delegate_to: claude
      reason: "Claude excels at code analysis"

    - trigger:
        keywords: [document, analyze, summarize]
        context_size: large
      delegate_to: pulse
      reason: "Pulse has 1M token context"

  behavior:
    explain_delegation: true    # Always explain why delegating
    prefer_offline: true         # Handle offline when possible
    privacy_warnings: true       # Warn before sending data online
    fallback_to_offline: true    # Use offline if online fails
```

### Workspace Configuration

**File:** `config/workspace.yaml`

```yaml
routing:
  # Avatar is the default - handles all requests
  default_backend: ollama_small

backends:
  # Offline avatar (primary)
  ollama_small:
    enabled: true
    provider: ollama
    model: mistral  # Or mixtral for more power

  # Online specialists (delegated when needed)
  claude:
    enabled: true
    provider: claude

  nova:
    enabled: true
    provider: openai

  pulse:
    enabled: true
    provider: gemini
```

## Using the Avatar

### Method 1: Default Behavior

Just send requests - avatar handles routing automatically:

```bash
curl -X POST http://localhost:8000/api/send \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain how TCP works",
    "context": {
      "pack": "avatar-orchestrator"
    }
  }'
```

The avatar will:
1. Assess the query complexity
2. Decide: handle offline or delegate
3. Explain its decision
4. Return the response

### Method 2: Explicit Avatar Context

Load the avatar context pack:

```bash
curl -X POST http://localhost:8000/api/context-packs/apply \
  -H "Content-Type: application/json" \
  -d '{
    "pack_name": "avatar-orchestrator",
    "prompt": "Review this code for bugs"
  }'
```

### Method 3: Force Offline

Use the `#offline` tag:

```bash
curl -X POST http://localhost:8000/api/send \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "#offline What is machine learning?"
  }'
```

This guarantees the response is handled entirely offline.

### Method 4: Direct Backend (Bypass Avatar)

Explicitly specify a backend:

```bash
curl -X POST http://localhost:8000/api/send \
  -H "Content-Type: application/json" \
  -d '{
    "backend": "claude",
    "prompt": "Write a sorting algorithm"
  }'
```

This bypasses the avatar and goes directly to Claude.

## Example Interactions

### Example 1: Simple Question (Offline)

**Request:**
```json
{
  "prompt": "What is the capital of France?"
}
```

**Response:**
```
Avatar (offline): "The capital of France is Paris.

I handled this myself since it's a straightforward geography question
that doesn't require specialized AI capabilities. This was instant
and kept your query private on your machine."
```

### Example 2: Coding Task (Delegated to Claude)

**Request:**
```json
{
  "prompt": "Write a Python function to find prime numbers with error handling"
}
```

**Response:**
```
Avatar: "This is a coding task requiring clean, production-ready code.
I'm delegating to Claude because:

- Claude specializes in code generation
- It provides error handling and type hints
- It follows best practices (PEP 8 for Python)

[Delegating to Claude...]

Here's Claude's response:

[Code implementation from Claude follows...]
```

### Example 3: Large Document Analysis (Delegated to Pulse)

**Request:**
```json
{
  "prompt": "Summarize this 50-page research paper",
  "context": {
    "document": "[50 pages of text...]"
  }
}
```

**Response:**
```
Avatar: "This document is too large for my context window (8K tokens).
I'm delegating to Pulse (Google Gemini) because:

- Pulse has a 1M token context window
- It excels at document analysis and summarization
- Perfect for processing long-form content

[Delegating to Pulse...]

Here's Pulse's summary:

[Summary follows...]
```

### Example 4: Privacy-Sensitive (Offline Only)

**Request:**
```json
{
  "prompt": "Help me remember my password: MyPassword123"
}
```

**Response:**
```
Avatar (offline): "⚠️ PRIVACY WARNING: I detected sensitive data
(a password). I'm handling this entirely offline for your security.

I'll NEVER send credentials to online AIs.

To remember passwords, I recommend using a password manager like:
- Bitwarden (open source)
- 1Password
- KeePassXC (offline)

This response was generated completely offline - your password
never left your machine."
```

### Example 5: Multi-AI Comparison

**Request:**
```json
{
  "prompt": "#multiverse What's the best database for a social media app?"
}
```

**Response:**
```
Avatar: "This question benefits from multiple expert perspectives.
I'm consulting Claude, Nova, and Pulse in parallel...

[Delegating to 3 AIs...]

**Claude recommends:** PostgreSQL
- Excellent ACID compliance
- Mature, battle-tested
- Great for relational data

**Nova recommends:** MongoDB
- Flexible schema
- Scales horizontally
- Good for rapid iteration

**Pulse recommends:** Hybrid approach
- PostgreSQL for user data (relational)
- Redis for caching
- Elasticsearch for search

**My synthesis:** Choose based on your team's expertise and scale.
Start with PostgreSQL for reliability, add MongoDB if you need
flexibility, or use the hybrid approach for optimal performance.
```

## Privacy Features

### Automatic Privacy Detection

The avatar automatically detects sensitive data:

```python
sensitive_keywords = [
    "password", "secret", "api_key", "token",
    "credential", "private", "confidential",
    "ssn", "credit card", "personal"
]
```

If detected → **OFFLINE ONLY**

### Privacy Levels

**Level 1: Public** → Can delegate to online AIs
- General knowledge questions
- Public information
- Code examples (non-proprietary)

**Level 2: Semi-Private** → Avatar asks permission
- Work-related questions
- Company-specific data
- Internal processes

**Level 3: Private** → **OFFLINE ONLY**
- Personal information
- Credentials (passwords, keys, tokens)
- Private files
- Sensitive data

### Override Privacy Settings

Force offline mode:

```bash
# Method 1: Use #private tag
curl -X POST http://localhost:8000/api/send \
  -d '{"prompt": "#private Help with my database schema"}'

# Method 2: Explicit backend
curl -X POST http://localhost:8000/api/send \
  -d '{"backend": "ollama_small", "prompt": "..."}'
```

## Performance Optimization

### When to Use Which AI

| Scenario | Use | Latency | Quality | Cost |
|----------|-----|---------|---------|------|
| Simple queries | Avatar (offline) | ~100ms | Good | Free |
| Coding | Claude | ~2s | Excellent | $$$ |
| General tasks | Nova | ~1s | Very Good | $$ |
| Large documents | Pulse | ~3s | Excellent | $ |
| Alternative view | Ara | ~10s | Good | Free |

### Caching

The avatar caches responses to avoid redundant online API calls:

```yaml
avatar:
  behavior:
    cache_responses: true
    cache_ttl: 3600  # 1 hour
```

### Fallback Strategy

If online AI fails:
1. Avatar attempts offline handling
2. Provides partial answer with disclaimer
3. Suggests retry or alternative AI

```
Avatar: "Claude is currently unavailable. I'll handle this offline,
         but my code analysis may be less detailed. Would you like me
         to try Nova instead?"
```

## Cost Savings

### Example Monthly Costs

**Without Avatar (All Online):**
- 1000 queries/month to GPT-4: ~$50
- 500 code reviews to Claude: ~$75
- Total: **$125/month**

**With Avatar (Hybrid):**
- 700 queries offline (avatar): **$0**
- 200 queries to online AIs: ~$25
- 100 code reviews to Claude: ~$15
- Total: **$40/month**

**Savings: 68% reduction**

## Troubleshooting

### Issue: Avatar always delegates (not using offline)

**Cause:** Avatar thinks all queries are complex

**Fix:** Adjust delegation threshold:

```yaml
avatar:
  behavior:
    delegation_threshold: high  # medium → high
```

### Issue: Avatar refuses to delegate (always offline)

**Cause:** `prefer_offline: true` is too aggressive

**Fix:**

```yaml
avatar:
  behavior:
    prefer_offline: false
```

### Issue: Privacy warnings for non-sensitive data

**Cause:** False positive on keyword detection

**Fix:** Update avatar.yaml:

```yaml
privacy:
  sensitive_keywords:
    # Remove overly broad keywords
    - password
    - api_key
    # Keep specific ones
```

### Issue: Slow avatar responses

**Cause:** Large Mixtral model on low RAM

**Fix:** Switch to smaller model:

```yaml
backends:
  ollama_small:
    model: mistral  # Instead of mixtral
```

## Advanced Usage

### Custom Delegation Rules

Add custom rules to `config/avatar.yaml`:

```yaml
delegation_rules:
  # Custom rule: Always use Claude for FastAPI
  - trigger:
      keywords: [fastapi, uvicorn, pydantic]
      type: coding
    delegate_to: claude
    reason: "Claude is excellent with FastAPI"

  # Custom rule: Use Nova for storytelling
  - trigger:
      keywords: [story, narrative, character]
      type: creative
    delegate_to: nova
    reason: "Nova excels at creative writing"
```

### Multi-Step Orchestration

The avatar can chain multiple AIs:

```
User: "Plan a feature, then implement it"

Avatar:
1. Asking Pulse to create a plan (good at planning)...
2. Passing plan to Claude for implementation (coding expert)...
3. Using Nova to write user-facing documentation...

[Results synthesized...]
```

Configure in `avatar.yaml`:

```yaml
delegation_rules:
  - trigger:
      keywords: [plan and implement, design and code]
      type: multi_step
    delegate_to:
      - backend: pulse
        task: planning
      - backend: claude
        task: implementation
    strategy: sequential
```

## Best Practices

### 1. Start with Avatar

Always let the avatar handle routing:
```bash
# Good
curl -X POST /api/send -d '{"prompt": "..."}'

# Also good (explicit avatar)
curl -X POST /api/send -d '{"backend": "ollama_small", ...}'
```

### 2. Trust the Avatar's Decisions

The avatar is configured to delegate appropriately. Don't override unless necessary.

### 3. Use Tags for Hints

Help the avatar with routing tags:
- `#offline` - Force offline
- `#code` - Hint at coding task
- `#multiverse` - Get multiple perspectives

### 4. Protect Privacy

Use `#private` tag for sensitive data:
```bash
curl -X POST /api/send -d '{"prompt": "#private My database password is..."}'
```

### 5. Monitor Costs

Check delegation stats:
```bash
curl http://localhost:8000/api/stats/delegations
```

## Summary

The **Avatar Orchestrator** provides:

✅ **Offline-first** - Privacy, speed, always available
✅ **Online when needed** - Access to specialized AIs
✅ **Transparent** - Explains delegation decisions
✅ **Privacy-focused** - Sensitive data stays offline
✅ **Cost-effective** - Only use expensive APIs when necessary
✅ **Flexible** - Easy to configure and customize

**Architecture in One Line:**
> Stable offline avatar + specialized online AIs = Best of both worlds

---

**Next Steps:**
1. Install Ollama: `docs/OLLAMA_SETUP.md`
2. Configure avatar: `config/avatar.yaml`
3. Start the server: `python -m src.ui.app`
4. Send your first query and watch the avatar orchestrate!

For more information, see:
- `docs/V1_SUMMARY.md` - Complete feature overview
- `docs/OLLAMA_SETUP.md` - Offline AI setup
- `config/avatar.yaml` - Avatar configuration reference
