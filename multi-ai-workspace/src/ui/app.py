"""FastAPI web application for Multi-AI Workspace.

Provides a web-based chat interface for interacting with multiple AI backends
through the routing system.
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio
from pathlib import Path

from ..core.router import Router
from ..core.backend import Context
from ..utils.config import load_config
from ..utils.logger import setup_logger, get_logger

# Initialize logger
setup_logger("INFO")
logger = get_logger(__name__)

# FastAPI app
app = FastAPI(
    title="Multi-AI Workspace",
    description="Intelligent multi-AI orchestration platform",
    version="0.1.0"
)

# Global router instance
router: Optional[Router] = None


class ChatRequest(BaseModel):
    """Chat request model."""
    message: str
    tags: Optional[List[str]] = None
    backend: Optional[str] = None
    stream: bool = False
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """Chat response model."""
    responses: List[Dict[str, Any]]
    routing: Dict[str, Any]


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    global router

    # Load configuration
    config_path = Path("config/workspace.yaml")

    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using minimal config")
        # Create minimal router for demo
        from ..integrations.pulse_backend import PulseBackend
        router = Router(
            backends={"pulse": PulseBackend()},
            default_backend="pulse"
        )
    else:
        try:
            config_loader = load_config(config_path)
            backends = config_loader.load_backends()
            rules = config_loader.load_routing_rules()
            default_backend = config_loader.get_default_backend()

            router = Router(
                backends=backends,
                rules=rules,
                default_backend=default_backend
            )

            logger.info("Router initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize router: {e}")
            raise

    # Health check all backends
    for name, backend in router.backends.items():
        try:
            healthy = await backend.health_check()
            status = "✅ healthy" if healthy else "❌ unhealthy"
            logger.info(f"Backend '{name}': {status}")
        except Exception as e:
            logger.error(f"Backend '{name}' health check failed: {e}")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main chat interface."""
    return get_chat_html()


@app.get("/api/backends")
async def list_backends():
    """List all available backends."""
    if not router:
        raise HTTPException(status_code=500, detail="Router not initialized")

    return JSONResponse({
        "backends": router.list_backends(),
        "routing_info": router.get_routing_info()
    })


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a chat message and get response(s).

    Args:
        request: Chat request with message and options

    Returns:
        ChatResponse with responses from selected backend(s)
    """
    if not router:
        raise HTTPException(status_code=500, detail="Router not initialized")

    logger.info(f"Chat request: {request.message[:50]}...")

    try:
        # Build context
        context = None
        if request.context:
            context = Context(**request.context)

        # Execute via router
        responses = await router.execute(
            prompt=request.message,
            tags=request.tags,
            context=context,
            force_backend=request.backend
        )

        # Get routing info
        routing_result = router.route(
            prompt=request.message,
            tags=request.tags,
            force_backend=request.backend
        )

        # Format responses
        formatted_responses = [
            {
                "backend": router.backends[name].name
                if name in router.backends else "unknown",
                "content": response.content,
                "provider": response.provider.value,
                "model": response.model,
                "latency_ms": response.latency_ms,
                "tokens_used": response.tokens_used,
                "error": response.error
            }
            for name, response in zip(
                [b.name for b in routing_result.backends],
                responses
            )
        ]

        return ChatResponse(
            responses=formatted_responses,
            routing={
                "strategy": routing_result.strategy.value,
                "backends": [b.name for b in routing_result.backends],
                "matched_rule": routing_result.matched_rule.tags
                if routing_result.matched_rule else None
            }
        )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for streaming chat.

    Allows real-time streaming of AI responses.
    """
    await websocket.accept()

    if not router:
        await websocket.send_json({"error": "Router not initialized"})
        await websocket.close()
        return

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()

            message = data.get("message")
            tags = data.get("tags")
            backend_name = data.get("backend")

            if not message:
                await websocket.send_json({"error": "No message provided"})
                continue

            # Route to backend
            routing_result = router.route(
                prompt=message,
                tags=tags,
                force_backend=backend_name
            )

            # Send routing info
            await websocket.send_json({
                "type": "routing",
                "strategy": routing_result.strategy.value,
                "backends": [b.name for b in routing_result.backends]
            })

            # Stream from first backend (for now)
            backend = routing_result.backends[0]

            await websocket.send_json({
                "type": "start",
                "backend": backend.name,
                "provider": backend.provider.value
            })

            async for chunk in backend.stream_message(message):
                await websocket.send_json({
                    "type": "chunk",
                    "content": chunk
                })

            await websocket.send_json({
                "type": "end"
            })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({"error": str(e)})


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    if not router:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": "Router not initialized"}
        )

    # Check backend health
    backend_health = {}
    for name, backend in router.backends.items():
        try:
            healthy = await backend.health_check()
            backend_health[name] = "healthy" if healthy else "unhealthy"
        except Exception as e:
            backend_health[name] = f"error: {e}"

    all_healthy = all(status == "healthy" for status in backend_health.values())

    return JSONResponse({
        "status": "healthy" if all_healthy else "degraded",
        "backends": backend_health
    })


def get_chat_html() -> str:
    """Get the chat interface HTML."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-AI Workspace</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a1a;
            color: #e0e0e0;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .header {
            background: #2d2d2d;
            padding: 1rem 2rem;
            border-bottom: 1px solid #404040;
        }
        .header h1 {
            font-size: 1.5rem;
            font-weight: 600;
            color: #fff;
        }
        .header .subtitle {
            font-size: 0.875rem;
            color: #a0a0a0;
            margin-top: 0.25rem;
        }
        .main {
            flex: 1;
            display: flex;
            overflow: hidden;
        }
        .chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            max-width: 1200px;
            margin: 0 auto;
            width: 100%;
        }
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 2rem;
        }
        .message {
            margin-bottom: 1.5rem;
            animation: fadeIn 0.3s;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .message.user {
            text-align: right;
        }
        .message-content {
            display: inline-block;
            padding: 0.75rem 1rem;
            border-radius: 0.5rem;
            max-width: 70%;
            text-align: left;
        }
        .message.user .message-content {
            background: #0084ff;
            color: white;
        }
        .message.assistant .message-content {
            background: #2d2d2d;
            border: 1px solid #404040;
        }
        .message-meta {
            font-size: 0.75rem;
            color: #808080;
            margin-top: 0.25rem;
        }
        .input-container {
            padding: 1rem 2rem;
            background: #2d2d2d;
            border-top: 1px solid #404040;
        }
        .input-row {
            display: flex;
            gap: 1rem;
            align-items: center;
        }
        #messageInput {
            flex: 1;
            padding: 0.75rem 1rem;
            background: #1a1a1a;
            border: 1px solid #404040;
            border-radius: 0.5rem;
            color: #e0e0e0;
            font-size: 1rem;
        }
        #messageInput:focus {
            outline: none;
            border-color: #0084ff;
        }
        button {
            padding: 0.75rem 1.5rem;
            background: #0084ff;
            color: white;
            border: none;
            border-radius: 0.5rem;
            cursor: pointer;
            font-size: 1rem;
            transition: background 0.2s;
        }
        button:hover {
            background: #0073e6;
        }
        button:disabled {
            background: #404040;
            cursor: not-allowed;
        }
        .loading {
            display: inline-block;
            width: 1rem;
            height: 1rem;
            border: 2px solid #404040;
            border-top-color: #0084ff;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Multi-AI Workspace</h1>
        <div class="subtitle">Intelligent multi-AI orchestration platform</div>
    </div>

    <div class="main">
        <div class="chat-container">
            <div class="messages" id="messages">
                <div class="message assistant">
                    <div class="message-content">
                        Welcome to Multi-AI Workspace! Send a message to get started.<br><br>
                        <strong>Tip:</strong> Use #tags to route to specific AIs (e.g., #fast, #creative, #code)
                    </div>
                </div>
            </div>

            <div class="input-container">
                <div class="input-row">
                    <input type="text" id="messageInput" placeholder="Type your message... (use #tags for routing)"
                           autocomplete="off">
                    <button id="sendButton" onclick="sendMessage()">Send</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        const messagesDiv = document.getElementById('messages');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');

        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });

        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;

            // Disable input
            messageInput.disabled = true;
            sendButton.disabled = true;

            // Add user message
            addMessage(message, 'user');
            messageInput.value = '';

            // Show loading
            const loadingDiv = addMessage('<div class="loading"></div>', 'assistant');

            try {
                // Send request
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });

                const data = await response.json();

                // Remove loading
                loadingDiv.remove();

                // Add responses
                for (const resp of data.responses) {
                    const meta = `${resp.backend} • ${resp.latency_ms?.toFixed(0)}ms`;
                    addMessage(resp.content, 'assistant', meta);
                }

            } catch (error) {
                loadingDiv.remove();
                addMessage(`Error: ${error.message}`, 'assistant');
            }

            // Re-enable input
            messageInput.disabled = false;
            sendButton.disabled = false;
            messageInput.focus();
        }

        function addMessage(content, role, meta = null) {
            const div = document.createElement('div');
            div.className = `message ${role}`;

            div.innerHTML = `
                <div class="message-content">${content}</div>
                ${meta ? `<div class="message-meta">${meta}</div>` : ''}
            `;

            messagesDiv.appendChild(div);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;

            return div;
        }
    </script>
</body>
</html>
    """


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
