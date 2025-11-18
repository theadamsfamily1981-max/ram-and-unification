# Installation Guide - Multi-AI Workspace

Complete installation instructions for Multi-AI Workspace Phase 1 (v0.1).

## System Requirements

### Required
- **Python**: 3.10 or higher
- **pip**: Latest version
- **Operating System**: Linux, macOS, or Windows

### Optional
- **Ollama**: For local LLM support (Pulse backend)
- **API Keys**:
  - Anthropic API key (for Claude)
  - OpenAI API key (for ChatGPT/Nova)

## Installation Steps

### 1. Python Environment

**Check Python version:**
```bash
python --version  # Should be 3.10+
```

**Create virtual environment:**
```bash
cd multi-ai-workspace
python -m venv venv
```

**Activate virtual environment:**

Linux/macOS:
```bash
source venv/bin/activate
```

Windows:
```bash
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Verify installation:**
```bash
python -c "import fastapi, anthropic, openai; print('✅ Dependencies installed')"
```

### 3. Configure API Keys

**Copy environment template:**
```bash
cp .env.example .env
```

**Edit `.env` file:**
```bash
# Use your preferred editor
nano .env  # or vim, code, etc.
```

**Add your API keys:**
```bash
ANTHROPIC_API_KEY=sk-ant-your-key-here
OPENAI_API_KEY=sk-your-key-here
```

**Where to get API keys:**
- **Claude (Anthropic)**: https://console.anthropic.com/
- **OpenAI (ChatGPT)**: https://platform.openai.com/api-keys

### 4. Install Ollama (Optional)

For local LLM support (Pulse backend):

**Linux:**
```bash
curl https://ollama.ai/install.sh | sh
```

**macOS:**
```bash
brew install ollama
```

**Windows:**
Download from https://ollama.ai/download

**Start Ollama and pull model:**
```bash
ollama serve &
ollama pull llama3.2
```

**Verify Ollama:**
```bash
curl http://localhost:11434/api/tags
```

### 5. Configure Workspace

**Review configuration:**
```bash
cat config/workspace.yaml
```

**Customize as needed:**
- Enable/disable backends
- Modify routing rules
- Adjust system settings

**Example - Disable OpenAI if no key:**
```yaml
backends:
  nova:
    enabled: false  # Set to false if no OpenAI key
```

### 6. Start the Application

**Run the server:**
```bash
python -m uvicorn src.ui.app:app --reload --host 0.0.0.0 --port 8000
```

**Expected output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Router initialized with 3 backends, 6 rules
INFO:     Backend 'claude': ✅ healthy
INFO:     Backend 'nova': ✅ healthy
INFO:     Backend 'pulse': ✅ healthy
```

**Open in browser:**
```
http://localhost:8000
```

## Verification

### Test 1: Health Check

```bash
curl http://localhost:8000/api/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "backends": {
    "claude": "healthy",
    "nova": "healthy",
    "pulse": "healthy"
  }
}
```

### Test 2: List Backends

```bash
curl http://localhost:8000/api/backends
```

### Test 3: Send Chat Message

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "#fast What is 2+2?",
    "tags": ["fast"]
  }'
```

### Test 4: Web UI

1. Open http://localhost:8000
2. Type: `#fast Hello!`
3. Verify response from Pulse backend

## Troubleshooting

### Issue: ModuleNotFoundError

**Error:**
```
ModuleNotFoundError: No module named 'anthropic'
```

**Solution:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: Backend Unhealthy

**Error:**
```
Backend 'claude': ❌ unhealthy
```

**Solution:**
1. Check API key in `.env`
2. Verify internet connection
3. Check API key permissions at provider console

### Issue: Ollama Not Found

**Error:**
```
Backend 'pulse' health check failed: Connection refused
```

**Solution:**
```bash
# Start Ollama server
ollama serve

# Pull required model
ollama pull llama3.2

# Verify
curl http://localhost:11434/api/tags
```

### Issue: Port Already in Use

**Error:**
```
ERROR: Address already in use
```

**Solution:**
```bash
# Use different port
python -m uvicorn src.ui.app:app --reload --port 8001

# Or kill process using port 8000
# Linux/macOS:
lsof -ti:8000 | xargs kill -9

# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Issue: Config File Not Found

**Error:**
```
Config file not found: config/workspace.yaml
```

**Solution:**
```bash
# Check you're in the right directory
pwd  # Should be .../multi-ai-workspace

# Verify config file exists
ls config/workspace.yaml
```

## Directory Structure After Installation

```
multi-ai-workspace/
├── venv/                  # Virtual environment (created)
├── .env                   # API keys (created from .env.example)
├── config/
│   └── workspace.yaml     # Configuration
├── src/
│   ├── core/
│   ├── integrations/
│   ├── ui/
│   └── utils/
├── logs/                  # Logs (created automatically)
├── requirements.txt
└── README.md
```

## Next Steps

After successful installation:

1. **Read the README**: `cat README.md`
2. **Explore configuration**: Review `config/workspace.yaml`
3. **Try different tags**: Test routing with `#code`, `#creative`, `#multiverse`
4. **Check architecture**: Read `docs/ARCHITECTURE.md`

## Minimal Installation (Ollama Only)

If you don't have API keys, you can run with just Ollama:

**1. Install Ollama:**
```bash
curl https://ollama.ai/install.sh | sh
ollama pull llama3.2
```

**2. Disable API backends in config:**
```yaml
backends:
  claude:
    enabled: false
  nova:
    enabled: false
  pulse:
    enabled: true  # Only Pulse enabled
```

**3. Set default backend:**
```yaml
routing:
  default_backend: pulse
```

**4. Run:**
```bash
python -m uvicorn src.ui.app:app --reload
```

## Production Deployment (Future)

For production use (not yet recommended for v0.1):

```bash
# Install production WSGI server
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.ui.app:app --bind 0.0.0.0:8000
```

## Support

If you encounter issues not covered here:

1. Check the main [README.md](README.md)
2. Review [ARCHITECTURE.md](docs/ARCHITECTURE.md)
3. Check logs in `logs/workspace.log`
4. Contact project maintainer

---

**Installation complete!** 🎉

Start chatting at http://localhost:8000
