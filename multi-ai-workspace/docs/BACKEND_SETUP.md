# Backend Setup Guide - Multi-AI Workspace

Complete guide for setting up all 4 AI backends: **Pulse**, **Nova**, **Ara**, and **Claude**.

## Overview

Multi-AI Workspace supports 4 distinct AI backends:

| AI Name | Provider | Backend Type | API/Automation |
|---------|----------|--------------|----------------|
| **Claude** | Anthropic (Claude.ai) | API | ✅ Official API |
| **Nova** | OpenAI (ChatGPT) | API | ✅ Official API |
| **Pulse** | Google (Gemini) | API | ✅ Official API |
| **Ara** | X.AI (Grok) | Selenium | ⚠️ Browser Automation |

## Prerequisites

- Python 3.10+
- Chrome browser (for Ara/Grok)
- API keys for Claude, Nova, and Pulse

---

## 1. Claude (Anthropic) Setup

**Purpose:** Complex reasoning, coding, analysis

### Get API Key

1. Visit https://console.anthropic.com/
2. Sign up / Login
3. Navigate to API Keys
4. Create new key
5. Copy key (starts with `sk-ant-`)

### Configure

Add to `.env`:
```bash
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxx
```

Or in `config/workspace.yaml`:
```yaml
backends:
  claude:
    enabled: true
    provider: claude
    model: sonnet  # or opus, haiku
    api_key: your-key-here  # Or use ${ANTHROPIC_API_KEY}
```

### Test

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "#code Write a Python hello world", "backend": "claude"}'
```

---

## 2. Nova (OpenAI/ChatGPT) Setup

**Purpose:** General tasks, creative writing, fast responses

### Get API Key

1. Visit https://platform.openai.com/api-keys
2. Sign up / Login
3. Create new secret key
4. Copy key (starts with `sk-`)

### Configure

Add to `.env`:
```bash
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxx
```

Or in `config/workspace.yaml`:
```yaml
backends:
  nova:
    enabled: true
    provider: openai
    model: gpt-4o  # or gpt-4-turbo, gpt-3.5-turbo
    api_key: your-key-here
```

### Test

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "#creative Write a haiku", "backend": "nova"}'
```

---

## 3. Pulse (Google Gemini) Setup

**Purpose:** Internal orchestrator, planner, fast reasoning

### Get API Key

1. Visit https://makersuite.google.com/app/apikey
2. Sign in with Google account
3. Click "Create API Key"
4. Select/create Google Cloud project
5. Copy key

### Configure

Add to `.env`:
```bash
GOOGLE_API_KEY=xxxxxxxxxxxxxxxxxxxxx
# Or alternatively:
GEMINI_API_KEY=xxxxxxxxxxxxxxxxxxxxx
```

Or in `config/workspace.yaml`:
```yaml
backends:
  pulse:
    enabled: true
    provider: gemini
    model: gemini-1.5-flash  # or gemini-pro, gemini-1.5-pro
    api_key: your-key-here
```

### Models

- `gemini-pro` - Balanced (32K context)
- `gemini-1.5-pro` - Advanced (1M context, vision)
- `gemini-1.5-flash` - **Recommended** - Fast (1M context, vision)

### Test

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Plan a multi-step task", "backend": "pulse"}'
```

---

## 4. Ara (Grok/X.AI) Setup

**Purpose:** Alternative perspective, adversarial testing

**⚠️ Note:** Grok does NOT have a public API. We use Selenium browser automation.

### Prerequisites

1. **X.com account** with Grok access
2. **Chrome browser** installed
3. **ChromeDriver** (auto-installed via webdriver-manager)

### Configure

Add to `.env`:
```bash
X_USERNAME=your_x_username
X_PASSWORD=your_x_password
```

Or in `config/workspace.yaml`:
```yaml
backends:
  ara:
    enabled: true  # Set to true when ready
    provider: grok
    username: your_x_username
    password: your_x_password
    headless: true  # Run browser in background
```

### Important Notes

- **Slower than API backends** (browser automation overhead)
- **Requires X.com account** with Grok beta access
- **May break** if X.com UI changes
- **Not recommended for production** - use for testing/comparison only

### Test

```bash
# Enable ara in config first, then:
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What makes you different from other AIs?", "backend": "ara"}'
```

---

## Multi-AI Features

### Compare All 4 AIs

```bash
curl -X POST http://localhost:8000/api/perspectives/compare \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain quantum computing in simple terms",
    "context": {"backends": ["claude", "nova", "pulse", "ara"]}
  }'
```

### Use #multiverse Tag

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "#multiverse What is the meaning of life?"}'
```

This automatically routes to all 4 AIs in parallel.

---

## Installation Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

New dependencies for all backends:
- `anthropic` - Claude
- `openai` - Nova
- `google-generativeai` - Pulse
- `selenium` + `webdriver-manager` - Ara

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Update Config

Edit `config/workspace.yaml`:
- Enable/disable backends
- Set models
- Configure routing rules

### 4. Start Server

```bash
python -m uvicorn src.ui.app:app --reload
```

### 5. Verify Backends

```bash
curl http://localhost:8000/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "backends": {
    "claude": "healthy",
    "nova": "healthy",
    "pulse": "healthy",
    "ara": "healthy"  // if enabled
  }
}
```

---

## Troubleshooting

### Claude: Invalid API Key

```
Error: Invalid API key
```

**Solution:**
- Check key starts with `sk-ant-`
- Verify key is active in console.anthropic.com
- Ensure no extra spaces in `.env`

### Nova: Authentication Failed

```
Error: Incorrect API key provided
```

**Solution:**
- Check key starts with `sk-`
- Verify account has credits at platform.openai.com/account/billing
- Try regenerating key

### Pulse: API Not Enabled

```
Error: API key not valid. Please pass a valid API key.
```

**Solution:**
- Enable Generative Language API in Google Cloud Console
- Ensure API key is for correct project
- Wait a few minutes after creating key

### Ara: Login Failed

```
Error: Login failed: No such element
```

**Solution:**
- Verify X.com credentials are correct
- Check if X.com requires 2FA (not supported yet)
- Try disabling headless mode: `headless: false` in config
- Update Chrome to latest version

### Ara: ChromeDriver Not Found

```
Error: Message: 'chromedriver' executable needs to be in PATH
```

**Solution:**
```bash
pip install --upgrade webdriver-manager
```

This auto-downloads correct ChromeDriver.

---

## Cost Comparison

| Backend | Pricing | Free Tier | Best For |
|---------|---------|-----------|----------|
| **Claude** | $0.003-0.015/1K tokens | ✅ Credits | Complex tasks |
| **Nova** | $0.001-0.005/1K tokens | ❌ | General use |
| **Pulse** | FREE | ✅ Free | Orchestration |
| **Ara** | Unknown | ❌ | Testing |

**Recommendation for cost-conscious users:**
- Use **Pulse (Gemini)** for most tasks (free)
- Use **Claude** for complex coding/reasoning
- Use **Nova** for creative writing
- Use **Ara** sparingly (slowest, least reliable)

---

## Backend Selection Guide

### When to Use Claude
- Complex code refactoring
- Security reviews
- Deep technical analysis
- Multi-step reasoning

### When to Use Nova
- Creative writing
- Quick explanations
- General knowledge queries
- Conversational responses

### When to Use Pulse
- Planning and orchestration
- Fast iterations
- Free tier tasks
- Experimentation

### When to Use Ara
- Alternative perspectives
- Adversarial testing
- Comparison studies
- Special Grok features (if any)

---

## Advanced Configuration

### Routing by Task Type

Edit `config/workspace.yaml`:

```yaml
routing:
  rules:
    # Code tasks → Claude
    - tags: [code, coding, debug]
      backends: [claude]
      strategy: single

    # Creative → Nova
    - tags: [creative, write]
      backends: [nova]
      strategy: single

    # Planning → Pulse
    - tags: [plan, orchestrate]
      backends: [pulse]
      strategy: single

    # Comparison → All 4
    - tags: [multiverse, compare]
      backends: [claude, nova, pulse, ara]
      strategy: parallel
```

### Sequential Processing

Pulse → Claude chain for iterative refinement:

```yaml
- tags: [chain, refine]
  backends: [pulse, claude]
  strategy: sequential
```

**Usage:**
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "#chain Design a REST API for user management"}'
```

Pulse drafts, Claude refines!

---

## Next Steps

- [Phase 2 Features](PHASE2_FEATURES.md) - Context Packs, Perspectives Mixer
- [API Documentation](../README.md) - Full API reference
- [Architecture](ARCHITECTURE.md) - System design

---

**All 4 backends configured!** You now have the complete Multi-AI Workspace v1 with **Pulse**, **Nova**, **Ara**, and **Claude**. 🚀
