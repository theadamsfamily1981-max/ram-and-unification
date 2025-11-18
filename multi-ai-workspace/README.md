# Multi-AI Workspace

**Intelligent multi-AI orchestration platform** for seamless collaboration between Claude, ChatGPT, Grok, and local LLMs.

## Overview

Multi-AI Workspace is a local orchestration platform that enables you to:

- **Route prompts intelligently** to the best AI for each task using tag-based routing
- **Compare perspectives** from multiple AIs simultaneously
- **Chain reasoning** by passing outputs between different AIs
- **Maintain privacy** with local LLM support (Ollama)
- **Reduce costs** by automatically selecting appropriate models

### Phase 1 (v0.1) - Core Infrastructure ✅

Current implementation includes:

- ✅ AI Backend abstraction (Claude, Nova/ChatGPT, Pulse/Ollama)
- ✅ Tag-based intelligent routing system
- ✅ Multiple collaboration strategies (single, parallel, sequential, competitive)
- ✅ FastAPI web interface with real-time chat
- ✅ YAML-based configuration
- ✅ Comprehensive logging and error handling

## Quick Start

### Prerequisites

- Python 3.10+
- (Optional) Ollama for local LLM support
- API keys for Claude and/or OpenAI

### Installation

```bash
# Clone the repository
cd multi-ai-workspace

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Configuration

Edit `config/workspace.yaml` to configure:

- **Backends**: Enable/disable AI providers, set models
- **Routing Rules**: Define tag-based routing logic
- **System Settings**: Logging, server, performance

### Running the Application

```bash
# Start the web server
python -m uvicorn src.ui.app:app --reload --host 0.0.0.0 --port 8000

# Open in browser
# http://localhost:8000
```

## Usage

### Tag-Based Routing

Use hashtags in your messages to route to specific AIs:

```
#code Fix this Python function
→ Routes to Claude (best for coding)

#fast What's 2+2?
→ Routes to Pulse (local, instant)

#creative Write a short story about AI
→ Routes to Nova (creative writing)

#multiverse What's the meaning of life?
→ Routes to ALL AIs for diverse perspectives
```

### Available Tags

| Tag | Behavior | Example |
|-----|----------|---------|
| `#fast`, `#quick` | Route to Pulse (local) | `#fast What's the weather?` |
| `#code`, `#debug` | Route to Claude | `#code Debug this function` |
| `#creative`, `#write` | Route to Nova | `#creative Write a poem` |
| `#multiverse` | Query all AIs | `#multiverse Best programming language?` |
| `#chain` | Sequential processing | `#chain Draft then refine this email` |

### API Endpoints

#### POST /api/chat

Send a chat message:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the best way to learn Python?",
    "tags": ["code"]
  }'
```

#### GET /api/backends

List available backends:

```bash
curl http://localhost:8000/api/backends
```

#### GET /api/health

Check system health:

```bash
curl http://localhost:8000/api/health
```

## Architecture

```
multi-ai-workspace/
├── src/
│   ├── core/              # Core abstractions
│   │   ├── backend.py     # AIBackend abstract class
│   │   └── router.py      # Tag-based routing system
│   ├── integrations/      # AI provider integrations
│   │   ├── claude_backend.py
│   │   ├── nova_backend.py
│   │   └── pulse_backend.py
│   ├── ui/                # Web interface
│   │   └── app.py         # FastAPI application
│   └── utils/             # Utilities
│       ├── config.py      # Configuration loader
│       └── logger.py      # Logging setup
├── config/
│   └── workspace.yaml     # Main configuration
├── docs/
│   └── ARCHITECTURE.md    # Detailed architecture
└── requirements.txt
```

## Configuration Examples

### Enable/Disable Backends

```yaml
backends:
  claude:
    enabled: true
    model: sonnet  # or opus, haiku

  nova:
    enabled: false  # Disable if no OpenAI key

  pulse:
    enabled: true
    model: llama3.2
```

### Custom Routing Rules

```yaml
routing:
  rules:
    - tags: [python, django]
      backends: [claude]
      strategy: single
      priority: 100

    - tags: [brainstorm]
      backends: [claude, nova, pulse]
      strategy: parallel
      priority: 90
```

## AI Backends

### Claude (Anthropic)

- **Best for**: Complex reasoning, coding, analysis
- **Models**: Opus, Sonnet, Haiku
- **Setup**: Set `ANTHROPIC_API_KEY` in `.env`

### Nova (OpenAI ChatGPT)

- **Best for**: General tasks, creative writing, fast responses
- **Models**: GPT-4o, GPT-4 Turbo, GPT-3.5
- **Setup**: Set `OPENAI_API_KEY` in `.env`

### Pulse (Local Ollama)

- **Best for**: Privacy, no cost, offline usage
- **Models**: Llama 3.2, Mistral, Phi-3
- **Setup**: Install Ollama and run `ollama pull llama3.2`

## Routing Strategies

### Single
Route to one best AI based on tags.

```yaml
strategy: single
backends: [claude]
```

### Parallel
Query multiple AIs simultaneously, return all responses.

```yaml
strategy: parallel
backends: [claude, nova, pulse]
```

### Sequential
Chain AIs together, passing output to next.

```yaml
strategy: sequential
backends: [pulse, claude]  # Pulse drafts, Claude refines
```

### Competitive
Query multiple AIs, return only the fastest/best response.

```yaml
strategy: competitive
backends: [claude, nova]
```

## Development

### Project Structure

```
src/
├── core/          # Core abstractions (AIBackend, Router)
├── integrations/  # AI provider implementations
├── ui/            # Web interface
└── utils/         # Configuration, logging
```

### Adding a New Backend

1. Implement `AIBackend` interface in `src/integrations/`
2. Add configuration in `config/workspace.yaml`
3. Register in `src/utils/config.py`

Example:

```python
from src.core.backend import AIBackend, Response

class MyBackend(AIBackend):
    async def send_message(self, prompt, context):
        # Implementation
        return Response(...)
```

## Roadmap

### Phase 1 (v0.1) - Core Infrastructure ✅
- [x] AIBackend abstraction
- [x] Claude, Nova, Pulse backends
- [x] Tag-based routing
- [x] FastAPI web UI
- [x] Configuration system

### Phase 2 (v0.2) - Collaboration Features
- [ ] Perspectives Mixer widget
- [ ] Response storage (SQLite)
- [ ] Cross-Posting Panel
- [ ] Context Packs

### Phase 3 (v0.3) - External Integrations
- [ ] GitHub Autopilot
- [ ] Colab Offload
- [ ] Research Scout

### Phase 4 (v0.4) - Advanced Features
- [ ] Voice Macros
- [ ] Google Hub
- [ ] Guardrail & Secret Scanner

## Troubleshooting

### Backend Not Available

```
Backend 'claude' health check failed
```

**Solution**: Check API key in `.env` and internet connection.

### Ollama Connection Error

```
Pulse health check failed: Connection refused
```

**Solution**: Start Ollama server:
```bash
ollama serve
ollama pull llama3.2
```

### Import Errors

```
ModuleNotFoundError: No module named 'anthropic'
```

**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

## Security & Privacy

- **API Keys**: Stored in `.env` (never committed to git)
- **Local LLMs**: Pulse backend runs locally via Ollama
- **Logging**: Sensitive data filtered from logs
- **CORS**: Configurable origins in `workspace.yaml`

## Contributing

This is currently a personal project. For bug reports or feature requests, please contact the maintainer.

## License

Proprietary - All rights reserved.

## Support

For questions or issues:
1. Check the [Architecture documentation](docs/ARCHITECTURE.md)
2. Review troubleshooting section above
3. Contact project maintainer

---

**Multi-AI Workspace v0.1** - Built with ❤️ for intelligent AI orchestration
