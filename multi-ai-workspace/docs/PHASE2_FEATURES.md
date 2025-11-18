# Phase 2 (v0.2) Features - Multi-AI Workspace

**Collaboration features for intelligent multi-AI interactions.**

Phase 2 introduces powerful collaboration tools that enable you to store conversations, compare AI perspectives, use pre-built contexts, and export responses in multiple formats.

## New Features

### 1. Response Storage System

SQLite-based persistent storage for all conversations and AI responses.

**Features:**
- Automatic conversation tracking
- Message history with full metadata
- Token usage and latency tracking
- Conversation statistics and analytics
- Context pack storage

**Database Schema:**
- `conversations` - Conversation metadata
- `messages` - All user and AI messages
- `context_packs` - Reusable context templates

**API Endpoints:**

```bash
# List recent conversations
GET /api/conversations?limit=50&offset=0

# Get specific conversation with messages
GET /api/conversations/{conversation_id}
```

**Example:**

```bash
curl http://localhost:8000/api/conversations

# Response:
{
  "conversations": [
    {
      "id": 1,
      "title": "Python async programming",
      "status": "active",
      "created_at": "2025-01-15T10:30:00",
      "updated_at": "2025-01-15T10:45:00"
    }
  ]
}
```

**Storage Location:**
- Database: `data/workspace.db`
- Automatic backup recommended

---

### 2. Perspectives Mixer

Compare responses from multiple AIs side-by-side to get diverse perspectives.

**Use Cases:**
- Compare coding approaches from different AIs
- Get multiple creative writing styles
- Validate technical information across models
- Find the best explanation for complex topics

**Features:**
- Parallel querying of multiple AIs
- Automatic analysis (fastest, longest, most concise)
- Response quality metrics
- Side-by-side comparison
- Voting support

**API Endpoint:**

```bash
POST /api/perspectives/compare
```

**Example:**

```python
import requests

response = requests.post("http://localhost:8000/api/perspectives/compare", json={
    "message": "Explain quantum computing",
    "context": {
        "backends": ["claude", "nova", "pulse"]
    }
})

data = response.json()

# Access comparison
for perspective in data["comparison"]["perspectives"]:
    print(f"{perspective['backend']}: {perspective['content'][:100]}...")

# View analysis
print(f"Fastest: {data['analysis']['recommendations']['fastest']}")
print(f"Most concise: {data['analysis']['recommendations']['most_concise']}")
```

**Analysis Metrics:**
- Length analysis (avg, min, max)
- Speed analysis (latency)
- Token usage
- Success/failure rates
- Recommendations (fastest, longest, most concise)

**Example Output:**

```json
{
  "comparison": {
    "prompt": "Explain quantum computing",
    "perspectives": [
      {
        "backend": "claude",
        "content": "Quantum computing harnesses...",
        "latency_ms": 1200,
        "tokens_used": 450
      },
      {
        "backend": "nova",
        "content": "Quantum computers use qubits...",
        "latency_ms": 800,
        "tokens_used": 320
      }
    ]
  },
  "analysis": {
    "recommendations": {
      "fastest": "nova",
      "longest_response": "claude",
      "most_concise": "nova"
    },
    "speed_analysis": {
      "avg_latency_ms": 1000
    }
  }
}
```

---

### 3. Context Packs

Pre-built context templates for common tasks, enabling quick setup with optimized system prompts.

**Built-in Packs:**

| Pack Name | Description | Default Backend | Tags |
|-----------|-------------|-----------------|------|
| `coding-python` | Python expert with best practices | claude | #code #python |
| `code-review` | Senior engineer code review | claude | #code #review |
| `creative-writer` | Creative writing assistant | nova | #creative #write |
| `technical-writer` | Technical documentation expert | claude | #documentation |
| `brainstorm` | Creative brainstorming | ALL | #creative #multiverse |
| `debugger` | Debug expert | claude | #code #debug |
| `eli5` | Explain Like I'm 5 | pulse | #fast |
| `research` | Research analyst | ALL | #multiverse |
| `interview-prep` | Technical interview coach | claude | #code |
| `quick-answer` | Fast, concise answers | pulse | #fast |

**API Endpoints:**

```bash
# List all packs
GET /api/context-packs

# Get specific pack
GET /api/context-packs/{pack_name}
```

**Example Usage:**

```python
import requests

# List available packs
response = requests.get("http://localhost:8000/api/context-packs")
packs = response.json()["packs"]

for pack in packs:
    print(f"{pack['name']}: {pack['description']}")

# Get specific pack
response = requests.get("http://localhost:8000/api/context-packs/coding-python")
pack = response.json()

print(f"System Prompt: {pack['system_prompt']}")
print(f"Default Tags: {pack['default_tags']}")
```

**Creating Custom Packs:**

```python
from src.widgets.context_packs import ContextPackManager
from src.storage.database import ResponseStore

store = ResponseStore()
manager = ContextPackManager(store)

# Create custom pack
pack = manager.create_custom_pack(
    name="my-react-expert",
    description="React and TypeScript expert",
    system_prompt="You are a React expert specializing in TypeScript, hooks, and modern best practices.",
    default_tags=["code", "react"],
    default_backend="claude",
    example_messages=[
        {"role": "user", "content": "How do I use useEffect?"},
        {"role": "assistant", "content": "useEffect is a React hook for side effects..."}
    ]
)
```

**Pack Features:**
- System prompts optimized for specific tasks
- Example conversations for few-shot learning
- Routing tags for automatic backend selection
- Usage tracking
- Custom pack creation

---

### 4. Cross-Posting Panel

Export AI responses in multiple formats for easy sharing.

**Supported Formats:**
- **text** - Plain text (`.txt`)
- **markdown** - Markdown with metadata (`.md`)
- **json** - Full JSON export (`.json`)
- **code** - Code snippets (`.txt`)
- **tweet** - Twitter/X format (280 char)
- **email** - Email draft format (`.txt`)

**API Endpoint:**

```bash
POST /api/export
```

**Example:**

```python
import requests

# Export response as Markdown
response = requests.post("http://localhost:8000/api/export", json={
    "content": "This is an AI-generated response about Python async programming...",
    "format_type": "markdown",
    "metadata": {
        "prompt": "Explain Python asyncio",
        "backend": "claude"
    }
})

result = response.json()

print(f"Exported to: {result['file_path']}")
print(f"Size: {result['size_bytes']} bytes")
```

**Export Conversation:**

```python
from src.widgets.cross_posting import CrossPostingPanel

panel = CrossPostingPanel("exports")

# Export entire conversation
messages = [
    {"role": "user", "content": "What is async?"},
    {"role": "assistant", "content": "Async enables...", "backend_name": "claude"},
    {"role": "user", "content": "Show example"},
    {"role": "assistant", "content": "Here's an example...", "backend_name": "claude"}
]

result = panel.export_conversation(
    messages=messages,
    format_type="markdown",
    title="Async Programming Discussion"
)

print(f"Conversation exported: {result['file_path']}")
```

**Export Directory:**
- Location: `exports/`
- Auto-created on first export
- Timestamped filenames

**Format Examples:**

**Markdown Export:**
```markdown
## Prompt

Explain Python asyncio

## Response

Python's asyncio module provides...

---
**Metadata**
- Backend: claude (claude-3-5-sonnet-20241022)
- Response time: 1200ms
- Tokens used: 450
- Generated: 2025-01-15 10:30:00
```

**Tweet Format:**
```
Python's asyncio enables concurrent code execution using async/await syntax. It's perfect for I/O-bound operations like web requests and file handling. Key concepts: event loop, coroutines, tasks...
```

---

## Usage Examples

### Example 1: Compare Coding Approaches

```python
import requests

# Compare how different AIs solve a coding problem
response = requests.post("http://localhost:8000/api/perspectives/compare", json={
    "message": "Write a function to find the longest palindrome in a string",
    "context": {
        "backends": ["claude", "nova"]
    }
})

data = response.json()

# Claude's approach
print("Claude:", data["comparison"]["perspectives"][0]["content"])

# Nova's approach
print("Nova:", data["comparison"]["perspectives"][1]["content"])

# Which was faster?
print("Fastest:", data["analysis"]["recommendations"]["fastest"])
```

### Example 2: Use Context Pack for Code Review

```python
# Apply code-review context pack
response = requests.post("http://localhost:8000/api/chat", json={
    "message": """
    Review this code:

    def process_data(data):
        result = []
        for item in data:
            if item > 0:
                result.append(item * 2)
        return result
    """,
    "context": {
        "pack": "code-review"  # Use code-review pack
    }
})

# Get detailed code review
print(response.json()["responses"][0]["content"])
```

### Example 3: Export Research Findings

```python
# Get research from multiple AIs
comparison = requests.post("http://localhost:8000/api/perspectives/compare", json={
    "message": "What are the pros and cons of microservices architecture?",
    "context": {
        "backends": ["claude", "nova", "pulse"]
    }
})

# Export as markdown report
for i, perspective in enumerate(comparison.json()["comparison"]["perspectives"]):
    requests.post("http://localhost:8000/api/export", json={
        "content": perspective["content"],
        "format_type": "markdown",
        "metadata": {
            "prompt": "Microservices pros and cons",
            "backend": perspective["backend"],
            "part": f"{i+1}/3"
        }
    })

print("Research exported to exports/")
```

---

## Database Schema

### Conversations Table

```sql
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY,
    title TEXT,
    status TEXT,  -- active, archived, deleted
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    metadata JSON
);
```

### Messages Table

```sql
CREATE TABLE messages (
    id INTEGER PRIMARY KEY,
    conversation_id INTEGER,
    role TEXT,  -- user, assistant, system
    content TEXT,
    backend_name TEXT,
    provider TEXT,
    model TEXT,
    tokens_used INTEGER,
    latency_ms REAL,
    tags JSON,
    routing_strategy TEXT,
    error TEXT,
    created_at TIMESTAMP,
    metadata JSON,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);
```

### Context Packs Table

```sql
CREATE TABLE context_packs (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE,
    description TEXT,
    system_prompt TEXT,
    example_messages JSON,
    default_tags JSON,
    default_backend TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    use_count INTEGER,
    metadata JSON
);
```

---

## API Reference

### Conversations

```
GET /api/conversations
GET /api/conversations/{id}
```

### Context Packs

```
GET /api/context-packs
GET /api/context-packs/{name}
```

### Perspectives

```
POST /api/perspectives/compare
```

### Export

```
POST /api/export
```

---

## Configuration

No additional configuration required for Phase 2! All features work with existing `config/workspace.yaml`.

**Optional Settings:**

```yaml
system:
  # Storage location
  database_path: data/workspace.db

  # Export directory
  export_dir: exports

  # Context packs
  enable_custom_packs: true
```

---

## Next Steps

### Phase 3 (v0.3) - Coming Soon

- GitHub Autopilot integration
- Google Colab offload for heavy workloads
- Research Scout for web research
- Enhanced UI with widget panels

---

## Troubleshooting

### Database Locked

**Issue:** `database is locked` error

**Solution:**
```bash
# Close all connections
pkill -f "uvicorn.*app:app"

# Restart server
python -m uvicorn src.ui.app:app --reload
```

### Export Directory Not Found

**Issue:** Export fails with directory error

**Solution:**
```python
from pathlib import Path
Path("exports").mkdir(exist_ok=True)
```

### Context Pack Not Found

**Issue:** Built-in pack not available

**Solution:**
Built-in packs are auto-created on first startup. Restart the server:
```bash
python -m uvicorn src.ui.app:app --reload
```

---

**Phase 2 Complete!** 🎉

You now have powerful collaboration tools for multi-AI workflows.
