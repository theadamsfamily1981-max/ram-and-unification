# Ollama Setup Guide - Offline AI Mode

This guide shows you how to set up **completely offline AI** using Ollama with Mistral and Mixtral models.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Available Models](#available-models)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)
- [Performance Tips](#performance-tips)

---

## Overview

**Ollama** enables you to run large language models (LLMs) locally on your machine:

✅ **Completely Offline** - No internet required after initial download
✅ **100% Private** - All data stays on your machine
✅ **Free** - No API costs or subscriptions
✅ **Fast** - No network latency
✅ **Flexible** - Switch between models instantly

**Recommended Models for Multi-AI Workspace:**
- **Mistral 7B** - Small, fast (4-8GB RAM)
- **Mixtral 8x7B** - Larger MoE model (24-32GB RAM)

---

## Installation

### Step 1: Install Ollama

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**macOS:**
```bash
# Using Homebrew
brew install ollama

# Or download from https://ollama.ai/download
```

**Windows:**
Download and install from: https://ollama.ai/download/windows

### Step 2: Verify Installation

```bash
ollama --version
```

Expected output: `ollama version 0.x.x`

### Step 3: Start Ollama Server

```bash
ollama serve
```

Leave this running in the background. The server runs on `http://localhost:11434` by default.

**Alternative (systemd on Linux):**
```bash
# Ollama typically installs as a systemd service
sudo systemctl status ollama
sudo systemctl start ollama
```

---

## Quick Start

### 1. Pull Mistral 7B (Small Model)

```bash
# Pull the small Mistral model (~4GB download)
ollama pull mistral:7b

# Or use the instruct-tuned version (better for tasks)
ollama pull mistral:7b-instruct
```

**Download size:** ~4GB
**RAM required:** 4-8GB
**Use case:** Fast offline AI for general tasks

### 2. Pull Mixtral 8x7B (Larger Model - Optional)

```bash
# Pull Mixtral MoE model (~26GB download)
ollama pull mixtral:8x7b

# Or quantized version for lower RAM usage
ollama pull mixtral:8x7b-instruct-v0.1-q4_0
```

**Download size:** ~26GB (full) or ~16GB (quantized)
**RAM required:** 24-32GB (full) or 16-24GB (quantized)
**Use case:** More powerful offline AI, better reasoning

### 3. Test the Model

```bash
# Test Mistral 7B
ollama run mistral:7b "Explain quantum computing in simple terms"

# Test Mixtral 8x7B
ollama run mixtral:8x7b "Write a Python function to sort a list"
```

### 4. Configure Multi-AI Workspace

Edit `config/workspace.yaml`:

```yaml
backends:
  # Enable Ollama Small (Mistral 7B)
  ollama_small:
    enabled: true  # Set to true
    provider: ollama
    model: mistral
    base_url: http://localhost:11434
    config:
      max_tokens: 2048
      temperature: 0.7

  # Enable Ollama Large (Mixtral 8x7B) - Optional
  ollama_large:
    enabled: true  # Set to true if you have 24GB+ RAM
    provider: ollama
    model: mixtral
    base_url: http://localhost:11434
    config:
      max_tokens: 4096
      temperature: 0.7
```

### 5. Start Multi-AI Workspace

```bash
cd /home/user/ram-and-unification/multi-ai-workspace
python -m src.ui.app

# Or with uvicorn
uvicorn src.ui.app:app --reload
```

### 6. Test Offline Mode

```bash
# Send message to Ollama backend
curl -X POST http://localhost:8000/api/send \
  -H "Content-Type: application/json" \
  -d '{
    "backend": "ollama_small",
    "prompt": "Write a hello world in Python"
  }'

# Use offline tag routing
curl -X POST http://localhost:8000/api/send \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "#offline Explain how a CPU works"
  }'
```

---

## Available Models

### Recommended Models

| Model | Size | RAM | Download | Speed | Quality | Best For |
|-------|------|-----|----------|-------|---------|----------|
| **mistral:7b** | 7B | 4-8GB | 4GB | ⚡⚡⚡ | ⭐⭐⭐ | General tasks, fast responses |
| **mistral:7b-instruct** | 7B | 4-8GB | 4GB | ⚡⚡⚡ | ⭐⭐⭐⭐ | Following instructions |
| **mixtral:8x7b** | 47B* | 24-32GB | 26GB | ⚡⚡ | ⭐⭐⭐⭐⭐ | Complex reasoning, coding |
| **mixtral:8x7b-q4_0** | 47B* | 16-24GB | 16GB | ⚡⚡ | ⭐⭐⭐⭐ | Lower RAM usage |
| **codestral:22b** | 22B | 16-20GB | 12GB | ⚡⚡ | ⭐⭐⭐⭐⭐ | Code generation |
| **llama3:8b** | 8B | 4-8GB | 4.7GB | ⚡⚡⚡ | ⭐⭐⭐⭐ | Alternative to Mistral |

*Mixtral is a Mixture of Experts (MoE) model with 47B total params but only ~13B active at a time

### Other Popular Models

```bash
# Llama 3 (Meta)
ollama pull llama3:8b
ollama pull llama3:70b  # Requires 64GB+ RAM

# Code-specialized
ollama pull codellama:7b
ollama pull codellama:13b

# Phi-3 (Microsoft - very small)
ollama pull phi3:mini  # Only 2GB!

# Gemma (Google)
ollama pull gemma:7b
```

### Check Available Models

```bash
# List all installed models
ollama list

# Get model info
ollama show mistral:7b
```

---

## Configuration

### Workspace Configuration

**File:** `config/workspace.yaml`

```yaml
backends:
  # Small offline model (default)
  ollama_small:
    enabled: true
    provider: ollama
    model: mistral  # Alias for mistral:7b
    base_url: http://localhost:11434
    config:
      max_tokens: 2048
      temperature: 0.7
      top_p: 0.9
      top_k: 40

  # Large offline model (optional)
  ollama_large:
    enabled: false  # Enable if you have 24GB+ RAM
    provider: ollama
    model: mixtral  # Alias for mixtral:8x7b
    base_url: http://localhost:11434
    config:
      max_tokens: 4096
      temperature: 0.7
      top_p: 0.9

  # Custom model example
  ollama_code:
    enabled: false
    provider: ollama
    model: codestral:22b  # Full model ID
    base_url: http://localhost:11434
    config:
      max_tokens: 4096
      temperature: 0.3  # Lower temp for code

routing:
  # Use Ollama as default for offline mode
  default_backend: ollama_small

  rules:
    # Offline routing
    - tags: [offline, local, private]
      backends: [ollama_small]
      strategy: single
      priority: 110
```

### Model Aliases

The following aliases are configured in `ollama_backend.py`:

| Alias | Model ID | Description |
|-------|----------|-------------|
| `mistral` | `mistral:7b` | Small Mistral model |
| `mistral-small` | `mistral:7b-instruct` | Instruct-tuned |
| `mixtral` | `mixtral:8x7b` | Large MoE model |
| `mixtral-instruct` | `mixtral:8x7b-instruct-v0.1-q4_0` | Quantized |
| `codestral` | `codestral:22b` | Code-specialized |
| `llama3` | `llama3:8b` | Llama 3 8B |

### Advanced Configuration

**Custom Ollama server location:**
```yaml
backends:
  ollama_remote:
    enabled: true
    provider: ollama
    model: mistral
    base_url: http://192.168.1.100:11434  # Remote server
```

**Multiple Ollama instances:**
```yaml
backends:
  ollama_cpu:
    provider: ollama
    model: mistral:7b
    base_url: http://localhost:11434

  ollama_gpu:
    provider: ollama
    model: mixtral:8x7b
    base_url: http://localhost:11435  # Different port
```

---

## Usage Examples

### Example 1: Basic Offline Query

```bash
curl -X POST http://localhost:8000/api/send \
  -H "Content-Type: application/json" \
  -d '{
    "backend": "ollama_small",
    "prompt": "Explain REST APIs in 3 sentences"
  }'
```

### Example 2: Offline Code Generation

```bash
curl -X POST http://localhost:8000/api/send \
  -H "Content-Type: application/json" \
  -d '{
    "backend": "ollama_small",
    "prompt": "#code Write a Python function to check if a string is a palindrome",
    "context": {
      "system_prompt": "You are a Python expert. Write clean, efficient code."
    }
  }'
```

### Example 3: Compare Small vs Large Models

```bash
curl -X POST http://localhost:8000/api/send-parallel \
  -H "Content-Type: application/json" \
  -d '{
    "backends": ["ollama_small", "ollama_large"],
    "prompt": "Explain the difference between TCP and UDP"
  }'
```

### Example 4: Offline GitHub Autopilot

```bash
# Generate commit message using Ollama
curl -X POST http://localhost:8000/api/github/commit-message \
  -H "Content-Type: application/json" \
  -d '{
    "backend": "ollama_small",
    "style": "conventional"
  }'
```

### Example 5: Streaming Response

```python
import httpx
import json

async def stream_ollama():
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "http://localhost:8000/api/stream",
            json={
                "backend": "ollama_small",
                "prompt": "Write a short story about AI"
            }
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    print(data.get("content", ""), end="", flush=True)
```

---

## Troubleshooting

### Issue: "Cannot connect to Ollama"

**Error:**
```
Cannot connect to Ollama at http://localhost:11434
```

**Solutions:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama server
ollama serve

# Or check systemd service (Linux)
sudo systemctl status ollama
sudo systemctl start ollama

# Check firewall
sudo ufw allow 11434
```

### Issue: "Model not found"

**Error:**
```
Model mistral:7b not found
```

**Solutions:**
```bash
# List installed models
ollama list

# Pull the model
ollama pull mistral:7b

# Check model name matches config
ollama show mistral:7b
```

### Issue: "Out of memory" or Slow Performance

**Symptoms:**
- System freezes
- Ollama crashes
- Very slow responses

**Solutions:**

**1. Use a smaller model:**
```bash
# Try Mistral 7B instead of Mixtral
ollama pull mistral:7b

# Or use quantized version
ollama pull mistral:7b-q4_0
```

**2. Use quantized models (lower precision, less RAM):**
```bash
# Q4 quantization (~4-bit)
ollama pull mixtral:8x7b-instruct-v0.1-q4_0

# Q5 quantization (better quality)
ollama pull mixtral:8x7b-q5_0
```

**3. Reduce max_tokens:**
```yaml
config:
  max_tokens: 1024  # Lower than 2048
```

**4. Check RAM usage:**
```bash
# Monitor RAM
htop

# Or
free -h
```

### Issue: Very Slow First Response

**Cause:** Model loading into memory (cold start)

**Solutions:**
```bash
# Pre-load model into memory
ollama run mistral:7b "hi"

# Keep Ollama running in background
ollama serve &

# Or use smaller models for faster loading
```

### Issue: Model Quality Poor

**Solutions:**

**1. Use instruct-tuned models:**
```bash
ollama pull mistral:7b-instruct  # Better than base mistral:7b
```

**2. Adjust temperature:**
```yaml
config:
  temperature: 0.3  # Lower = more focused, deterministic
  temperature: 0.9  # Higher = more creative, varied
```

**3. Use larger model if RAM allows:**
```bash
ollama pull mixtral:8x7b  # Much better quality
```

---

## Performance Tips

### 1. GPU Acceleration

**Ollama automatically uses GPU if available (NVIDIA CUDA, Apple Metal)**

**Check GPU usage:**
```bash
# NVIDIA
nvidia-smi

# Apple Silicon (M1/M2/M3)
# Ollama automatically uses Metal

# AMD (ROCm)
rocm-smi
```

**Force CPU-only (if needed):**
```bash
OLLAMA_NUM_GPU=0 ollama serve
```

### 2. Model Selection by Use Case

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| Fast offline queries | `mistral:7b` | Best speed/quality ratio |
| Code generation | `codestral:22b` | Specialized for code |
| Complex reasoning | `mixtral:8x7b` | Highest quality |
| Low RAM systems | `phi3:mini` | Only 2GB RAM needed |
| Following instructions | `mistral:7b-instruct` | Instruction-tuned |

### 3. Optimize Configuration

**For speed:**
```yaml
config:
  max_tokens: 1024  # Lower token limit
  temperature: 0.3  # More deterministic
  top_k: 20        # Smaller sampling
```

**For quality:**
```yaml
config:
  max_tokens: 4096  # Higher limit
  temperature: 0.7  # Balanced
  top_p: 0.95      # Larger nucleus sampling
```

**For creativity:**
```yaml
config:
  max_tokens: 2048
  temperature: 0.9  # More random
  top_p: 0.95
  top_k: 40
```

### 4. Pre-load Models

```bash
# Keep models warm in memory
ollama run mistral:7b "ping" &
ollama run mixtral:8x7b "ping" &
```

### 5. Batch Requests

Use parallel routing to compare models efficiently:
```bash
curl -X POST http://localhost:8000/api/send-parallel \
  -d '{"backends": ["ollama_small", "claude"], "prompt": "test"}'
```

---

## Comparison: Ollama vs Cloud APIs

| Feature | Ollama (Offline) | Claude/GPT (Online) |
|---------|-----------------|---------------------|
| **Cost** | Free | $0.001-0.015 per 1K tokens |
| **Privacy** | 100% local | Sent to cloud |
| **Speed** | Fast (no network) | Network latency |
| **Quality** | Good (Mixtral) | Excellent (GPT-4, Claude) |
| **Context** | 8K-32K tokens | 128K-200K tokens |
| **Internet** | Not required | Required |
| **Setup** | Manual | API key only |
| **RAM** | 4-32GB | None |

---

## Quick Reference

### Essential Commands

```bash
# Server
ollama serve                    # Start server
ollama list                     # List models
ollama ps                       # Show running models

# Models
ollama pull mistral:7b          # Download model
ollama rm mistral:7b            # Remove model
ollama show mistral:7b          # Model info

# Testing
ollama run mistral:7b "prompt"  # Interactive
ollama run mistral:7b < file.txt # From file

# API
curl http://localhost:11434/api/tags  # List models (API)
```

### Configuration Files

- **Workspace config:** `config/workspace.yaml`
- **Ollama backend:** `src/integrations/ollama_backend.py`
- **Ollama data:** `~/.ollama/` (Linux/macOS) or `%USERPROFILE%\.ollama\` (Windows)

---

## Next Steps

1. **Install Ollama:** Follow installation steps above
2. **Pull Mistral 7B:** `ollama pull mistral:7b`
3. **Enable in config:** Set `enabled: true` in `workspace.yaml`
4. **Test:** Send a test request to verify offline mode works
5. **Optional:** Pull Mixtral 8x7B for better quality (if RAM allows)

For more help:
- Ollama docs: https://github.com/ollama/ollama
- Multi-AI Workspace docs: `docs/V1_SUMMARY.md`
- Troubleshooting: https://github.com/ollama/ollama/blob/main/docs/troubleshooting.md

---

**You're now ready to use Multi-AI Workspace completely offline! 🚀**
