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
import subprocess
import os
import shutil

from ..core.router import Router
from ..core.backend import Context
from ..utils.config import load_config
from ..utils.logger import setup_logger, get_logger
from ..storage.database import ResponseStore
from ..widgets.perspectives_mixer import PerspectivesMixer
from ..widgets.context_packs import ContextPackManager
from ..widgets.cross_posting import CrossPostingPanel
from ..widgets.github_autopilot import GitHubAutopilot
from ..widgets.colab_offload import ColabOffload

# Initialize logger
setup_logger("INFO")
logger = get_logger(__name__)

# FastAPI app
app = FastAPI(
    title="Multi-AI Workspace",
    description="Intelligent multi-AI orchestration platform - v1 Complete",
    version="1.0.0"
)

# Global instances
router: Optional[Router] = None
store: Optional[ResponseStore] = None
perspectives_mixer: Optional[PerspectivesMixer] = None
context_manager: Optional[ContextPackManager] = None
cross_posting: Optional[CrossPostingPanel] = None
github_autopilot: Optional[GitHubAutopilot] = None
colab_offload: Optional[ColabOffload] = None


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
    global router, store, perspectives_mixer, context_manager, cross_posting, github_autopilot, colab_offload

    # Initialize storage
    store = ResponseStore("data/workspace.db")
    logger.info("ResponseStore initialized")

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

    # Initialize all widgets
    perspectives_mixer = PerspectivesMixer(router, store)
    context_manager = ContextPackManager(store)
    cross_posting = CrossPostingPanel("exports")
    github_autopilot = GitHubAutopilot(router, repo_path=".", store=store)
    colab_offload = ColabOffload()

    logger.info("✅ v1 Complete: All widgets initialized")
    logger.info("   - Multi-AI Router (4 backends)")
    logger.info("   - Perspectives Mixer")
    logger.info("   - Context Packs")
    logger.info("   - Cross-Posting Panel")
    logger.info("   - GitHub Autopilot")
    logger.info("   - Colab Offload")


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
        "backends": backend_health,
        "version": "1.0.0",
        "phase": "v1 Complete",
        "features": {
            "multi_ai_router": ["pulse", "nova", "ara", "claude"],
            "perspectives_mixer": True,
            "context_packs": True,
            "cross_posting": True,
            "github_autopilot": True,
            "colab_offload": True,
            "response_storage": True
        }
    })


# ===== Phase 2 API Endpoints =====

@app.get("/api/context-packs")
async def list_context_packs():
    """List all available context packs."""
    if not context_manager:
        raise HTTPException(status_code=500, detail="Context manager not initialized")

    packs = context_manager.list_packs()

    return JSONResponse({
        "packs": [pack.to_dict() for pack in packs],
        "total": len(packs)
    })


@app.get("/api/context-packs/{pack_name}")
async def get_context_pack(pack_name: str):
    """Get a specific context pack."""
    if not context_manager:
        raise HTTPException(status_code=500, detail="Context manager not initialized")

    pack = context_manager.get_pack(pack_name)

    if not pack:
        raise HTTPException(status_code=404, detail=f"Pack not found: {pack_name}")

    return JSONResponse(pack.to_dict())


@app.post("/api/perspectives/compare")
async def compare_perspectives(request: ChatRequest):
    """Compare responses from multiple AIs."""
    if not perspectives_mixer:
        raise HTTPException(status_code=500, detail="Perspectives mixer not initialized")

    try:
        # Build context
        context = None
        if request.context:
            context = Context(**request.context)

        # Perform comparison
        comparison = await perspectives_mixer.compare(
            prompt=request.message,
            backends=request.context.get("backends") if request.context else None,
            context=context,
            save_to_store=True
        )

        # Analyze perspectives
        analysis = perspectives_mixer.analyze_perspectives(comparison)

        return JSONResponse({
            "comparison": comparison.to_dict(),
            "analysis": analysis
        })

    except Exception as e:
        logger.error(f"Perspectives comparison error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/conversations")
async def list_conversations(limit: int = 50, offset: int = 0):
    """List recent conversations."""
    if not store:
        raise HTTPException(status_code=500, detail="Storage not initialized")

    conversations = store.list_conversations(limit=limit, offset=offset)

    return JSONResponse({
        "conversations": [conv.to_dict() for conv in conversations],
        "limit": limit,
        "offset": offset
    })


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: int):
    """Get conversation with messages."""
    if not store:
        raise HTTPException(status_code=500, detail="Storage not initialized")

    conversation = store.get_conversation(conversation_id)

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = store.get_messages(conversation_id)

    return JSONResponse({
        "conversation": conversation.to_dict(),
        "messages": [msg.to_dict() for msg in messages],
        "stats": store.get_conversation_stats(conversation_id)
    })


@app.post("/api/export")
async def export_response(
    content: str,
    format_type: str = "text",
    metadata: Optional[Dict[str, Any]] = None
):
    """Export content in specified format."""
    if not cross_posting:
        raise HTTPException(status_code=500, detail="Cross-posting not initialized")

    # Create a mock Response object for export
    from ..core.backend import Response, AIProvider
    from datetime import datetime

    response = Response(
        content=content,
        provider=AIProvider.CUSTOM,
        model="unknown",
        metadata=metadata or {}
    )

    result = cross_posting.export_response(response, format_type)

    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Export failed"))

    return JSONResponse(result)


# ===== v1 Widget Endpoints: GitHub Autopilot & Colab Offload =====

@app.get("/api/github/status")
async def github_status():
    """Get current git repository status."""
    if not github_autopilot:
        raise HTTPException(status_code=500, detail="GitHub Autopilot not initialized")

    status = github_autopilot.get_status()
    return JSONResponse(status)


@app.get("/api/github/diff")
async def github_diff(file_path: Optional[str] = None, staged: bool = False):
    """Get git diff for changes."""
    if not github_autopilot:
        raise HTTPException(status_code=500, detail="GitHub Autopilot not initialized")

    diff = github_autopilot.get_diff(file_path, staged)
    return JSONResponse({"diff": diff})


@app.post("/api/github/explain")
async def github_explain(
    file_path: Optional[str] = None,
    backend: str = "claude"
):
    """AI explanation of git changes."""
    if not github_autopilot:
        raise HTTPException(status_code=500, detail="GitHub Autopilot not initialized")

    result = await github_autopilot.explain_changes(file_path, backend)
    return JSONResponse(result)


@app.post("/api/github/commit-message")
async def github_commit_message(
    backend: str = "claude",
    style: str = "conventional"
):
    """Generate commit message for staged changes."""
    if not github_autopilot:
        raise HTTPException(status_code=500, detail="GitHub Autopilot not initialized")

    result = await github_autopilot.generate_commit_message(backend, style)
    return JSONResponse(result)


@app.post("/api/github/review")
async def github_review(
    file_path: Optional[str] = None,
    backend: str = "claude",
    focus: Optional[List[str]] = None
):
    """AI code review of changes."""
    if not github_autopilot:
        raise HTTPException(status_code=500, detail="GitHub Autopilot not initialized")

    result = await github_autopilot.review_code(file_path, backend, focus)
    return JSONResponse(result)


@app.post("/api/github/pr-body")
async def github_pr_body(
    title: str,
    base_branch: str = "main",
    backend: str = "claude"
):
    """Generate Pull Request body."""
    if not github_autopilot:
        raise HTTPException(status_code=500, detail="GitHub Autopilot not initialized")

    result = await github_autopilot.generate_pr_body(title, base_branch, backend)
    return JSONResponse(result)


@app.get("/api/github/changes")
async def github_changes():
    """Get summary of file changes."""
    if not github_autopilot:
        raise HTTPException(status_code=500, detail="GitHub Autopilot not initialized")

    cards = github_autopilot.get_file_changes_summary()
    return JSONResponse({"changes": cards})


@app.get("/api/colab/info")
async def colab_info():
    """Get Colab Offload setup information."""
    if not colab_offload:
        raise HTTPException(status_code=500, detail="Colab Offload not initialized")

    info = colab_offload.get_offload_info()
    return JSONResponse(info)


@app.post("/api/colab/upload")
async def colab_upload(
    notebook_path: str,
    custom_name: Optional[str] = None
):
    """Upload notebook to Google Drive for Colab."""
    if not colab_offload:
        raise HTTPException(status_code=500, detail="Colab Offload not initialized")

    result = colab_offload.upload_notebook(notebook_path, custom_name)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return JSONResponse(result)


@app.post("/api/colab/upload-code")
async def colab_upload_code(
    code: str,
    filename: str = "generated_notebook.ipynb",
    language: str = "python"
):
    """Upload code as Colab notebook."""
    if not colab_offload:
        raise HTTPException(status_code=500, detail="Colab Offload not initialized")

    result = colab_offload.upload_code_as_notebook(code, filename, language)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return JSONResponse(result)


@app.get("/api/colab/files")
async def colab_files(limit: int = 20):
    """List uploaded Colab files."""
    if not colab_offload:
        raise HTTPException(status_code=500, detail="Colab Offload not initialized")

    files = colab_offload.list_uploaded_files(limit)
    return JSONResponse({"files": files})


# ===== System Setup & Control Endpoints =====

@app.get("/api/system/status")
async def system_status():
    """Get comprehensive system status and installation state."""
    home_dir = Path.home()
    base_dir = home_dir / "tfan-ara-system"

    # Check for components
    status = {
        "avatar_system": {
            "installed": Path("/home/user/ram-and-unification").exists(),
            "path": "/home/user/ram-and-unification"
        },
        "tfan_cockpit": {
            "installed": (base_dir / "Quanta-meis-nib-cis").exists(),
            "path": str(base_dir / "Quanta-meis-nib-cis")
        },
        "ollama": {
            "installed": shutil.which("ollama") is not None,
            "running": False
        },
        "dependencies": {
            "python": shutil.which("python3") is not None,
            "ffmpeg": shutil.which("ffmpeg") is not None,
            "git": shutil.which("git") is not None
        },
        "base_directory": str(base_dir),
        "home_directory": str(home_dir)
    }

    # Check if Ollama is running
    if status["ollama"]["installed"]:
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                timeout=5
            )
            status["ollama"]["running"] = result.returncode == 0
        except:
            pass

    return JSONResponse(status)


@app.post("/api/system/install-dependencies")
async def install_dependencies():
    """Install system dependencies."""
    try:
        script_path = Path("/home/user/ram-and-unification/install_dependencies.sh")

        if not script_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Installation script not found. Please ensure ram-and-unification is set up."
            )

        # Run the installer
        process = subprocess.Popen(
            ["bash", str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        output_lines = []
        for line in iter(process.stdout.readline, ''):
            if line:
                output_lines.append(line.strip())

        process.wait()

        return JSONResponse({
            "success": process.returncode == 0,
            "output": "\n".join(output_lines[-50:]),  # Last 50 lines
            "exit_code": process.returncode
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/system/install-tfan")
async def install_tfan():
    """Download and install T-FAN cockpit."""
    try:
        home_dir = Path.home()
        base_dir = home_dir / "tfan-ara-system"
        tfan_dir = base_dir / "Quanta-meis-nib-cis"

        # Create base directory
        base_dir.mkdir(parents=True, exist_ok=True)

        # Try to clone from GitHub
        try:
            result = subprocess.run(
                ["git", "clone",
                 "https://github.com/theadamsfamily1981-max/Quanta-meis-nib-cis",
                 str(tfan_dir)],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                return JSONResponse({
                    "success": True,
                    "message": "T-FAN downloaded successfully",
                    "path": str(tfan_dir)
                })
            else:
                # Try alternate locations
                alt_paths = [
                    home_dir / "Quanta-meis-nib-cis-main",
                    home_dir / "Quanta-meis-nib-cis",
                    Path("/home/user/Quanta-meis-nib-cis-main"),
                    Path("/home/user/Quanta-meis-nib-cis")
                ]

                for alt_path in alt_paths:
                    if alt_path.exists():
                        shutil.copytree(alt_path, tfan_dir, dirs_exist_ok=True)
                        return JSONResponse({
                            "success": True,
                            "message": f"T-FAN copied from {alt_path}",
                            "path": str(tfan_dir)
                        })

                return JSONResponse({
                    "success": False,
                    "message": "Could not download T-FAN. Please download manually from GitHub.",
                    "manual_url": "https://github.com/theadamsfamily1981-max/Quanta-meis-nib-cis",
                    "install_path": str(tfan_dir)
                })

        except subprocess.TimeoutExpired:
            return JSONResponse({
                "success": False,
                "message": "Download timed out. Please check your internet connection."
            })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/system/fix-webkit")
async def fix_webkit():
    """Fix WebKit compatibility issues."""
    try:
        script_path = Path("/home/user/ram-and-unification/fix_webkit_now.sh")

        if not script_path.exists():
            raise HTTPException(
                status_code=404,
                detail="WebKit fix script not found"
            )

        result = subprocess.run(
            ["bash", str(script_path)],
            capture_output=True,
            text=True,
            timeout=30
        )

        return JSONResponse({
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else None
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/system/setup-ara")
async def setup_ara():
    """Run complete Ara setup."""
    try:
        script_path = Path("/home/user/ram-and-unification/setup_ara.sh")

        if not script_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Ara setup script not found"
            )

        # Run setup in background
        process = subprocess.Popen(
            ["bash", str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            stdin=subprocess.PIPE
        )

        # Auto-answer prompts with 'y'
        process.stdin.write("y\ny\n")
        process.stdin.flush()

        output_lines = []
        for line in iter(process.stdout.readline, ''):
            if line:
                output_lines.append(line.strip())
                if len(output_lines) > 100:  # Limit output
                    break

        return JSONResponse({
            "success": True,
            "message": "Ara setup started",
            "output": "\n".join(output_lines[-30:])
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/system/launch/{component}")
async def launch_component(component: str):
    """Launch a system component."""
    try:
        commands = {
            "api": ["python3", "-m", "src.main"],
            "voice": ["python3", "ara_voice_interface.py"],
            "tfan": ["tfan-gnome"],
            "launcher": ["bash", "start_ara.sh"]
        }

        if component not in commands:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown component: {component}"
            )

        cmd = commands[component]
        cwd = Path("/home/user/ram-and-unification")

        # Launch in background
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True
        )

        return JSONResponse({
            "success": True,
            "message": f"{component} launched",
            "pid": process.pid
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/setup", response_class=HTMLResponse)
async def setup_page():
    """Serve the setup wizard page."""
    return get_setup_html()


def get_setup_html() -> str:
    """Get the setup wizard HTML."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ara Avatar System Setup</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem;
        }
        .wizard {
            background: white;
            border-radius: 1rem;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 800px;
            width: 100%;
            overflow: hidden;
        }
        .wizard-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            text-align: center;
        }
        .wizard-header h1 {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        .wizard-header p {
            opacity: 0.9;
            font-size: 1.1rem;
        }
        .wizard-body {
            padding: 2rem;
        }
        .step {
            display: none;
        }
        .step.active {
            display: block;
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin: 2rem 0;
        }
        .status-card {
            background: #f8f9fa;
            border: 2px solid #e9ecef;
            border-radius: 0.5rem;
            padding: 1.5rem;
        }
        .status-card.installed {
            border-color: #28a745;
            background: #d4edda;
        }
        .status-card.missing {
            border-color: #ffc107;
            background: #fff3cd;
        }
        .status-card h3 {
            font-size: 1rem;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .status-card p {
            font-size: 0.875rem;
            color: #666;
        }
        .status-icon {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            display: inline-block;
        }
        .status-icon.ok { background: #28a745; }
        .status-icon.warn { background: #ffc107; }
        .status-icon.error { background: #dc3545; }
        .btn {
            display: inline-block;
            padding: 0.75rem 2rem;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 0.5rem;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s;
            text-decoration: none;
        }
        .btn:hover {
            background: #5568d3;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        .btn-group {
            display: flex;
            gap: 1rem;
            margin-top: 2rem;
            justify-content: center;
        }
        .btn-secondary {
            background: #6c757d;
        }
        .btn-secondary:hover {
            background: #5a6268;
        }
        .btn-success {
            background: #28a745;
        }
        .btn-success:hover {
            background: #218838;
        }
        .progress-bar {
            background: #e9ecef;
            border-radius: 1rem;
            height: 8px;
            margin: 2rem 0;
            overflow: hidden;
        }
        .progress-fill {
            background: linear-gradient(90deg, #667eea, #764ba2);
            height: 100%;
            transition: width 0.3s;
        }
        .log-output {
            background: #1a1a1a;
            color: #0f0;
            font-family: 'Courier New', monospace;
            padding: 1rem;
            border-radius: 0.5rem;
            max-height: 300px;
            overflow-y: auto;
            font-size: 0.875rem;
            margin: 1rem 0;
        }
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .install-item {
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0.25rem;
        }
        .install-item h4 {
            margin-bottom: 0.5rem;
        }
        .install-item button {
            margin-top: 0.5rem;
        }
    </style>
</head>
<body>
    <div class="wizard">
        <div class="wizard-header">
            <h1>🚀 Ara Avatar System Setup</h1>
            <p>Complete installation and configuration wizard</p>
        </div>

        <div class="wizard-body">
            <!-- Step 1: System Check -->
            <div class="step active" id="step1">
                <h2>Step 1: System Status</h2>
                <p>Checking your system...</p>

                <div class="progress-bar">
                    <div class="progress-fill" style="width: 25%"></div>
                </div>

                <div class="status-grid" id="statusGrid">
                    <div class="status-card">
                        <h3><span class="loading"></span> Checking...</h3>
                    </div>
                </div>

                <div class="btn-group">
                    <button class="btn" onclick="checkSystem()">Refresh Status</button>
                    <button class="btn btn-success" onclick="goToStep(2)" id="continueBtn" disabled>Continue →</button>
                </div>
            </div>

            <!-- Step 2: Install Missing Components -->
            <div class="step" id="step2">
                <h2>Step 2: Install Components</h2>
                <p>Install missing components for full functionality</p>

                <div class="progress-bar">
                    <div class="progress-fill" style="width: 50%"></div>
                </div>

                <div id="installItems"></div>

                <div class="btn-group">
                    <button class="btn btn-secondary" onclick="goToStep(1)">← Back</button>
                    <button class="btn btn-success" onclick="goToStep(3)">Continue →</button>
                </div>
            </div>

            <!-- Step 3: Launch -->
            <div class="step" id="step3">
                <h2>Step 3: Ready to Launch!</h2>
                <p>Your Ara Avatar System is ready to use</p>

                <div class="progress-bar">
                    <div class="progress-fill" style="width: 100%"></div>
                </div>

                <div class="status-grid">
                    <div class="status-card installed">
                        <h3>🎤 Voice Interface</h3>
                        <p>Talk to Ara with voice commands</p>
                        <button class="btn" onclick="launch('voice')" style="margin-top: 1rem; font-size: 0.875rem;">Launch</button>
                    </div>
                    <div class="status-card installed">
                        <h3>🌐 Web Chat</h3>
                        <p>Multi-AI chat interface</p>
                        <button class="btn" onclick="goToChat()" style="margin-top: 1rem; font-size: 0.875rem;">Open Chat</button>
                    </div>
                    <div class="status-card installed">
                        <h3>🖥️ T-FAN Cockpit</h3>
                        <p>Spaceship-style HUD interface</p>
                        <button class="btn" onclick="launch('tfan')" style="margin-top: 1rem; font-size: 0.875rem;">Launch</button>
                    </div>
                    <div class="status-card installed">
                        <h3>🎬 Avatar Generator</h3>
                        <p>Create talking avatar videos</p>
                        <button class="btn" onclick="launch('api')" style="margin-top: 1rem; font-size: 0.875rem;">Start API</button>
                    </div>
                </div>

                <div class="btn-group">
                    <button class="btn btn-secondary" onclick="goToStep(1)">Run Setup Again</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        let systemStatus = {};

        async function checkSystem() {
            try {
                const response = await fetch('/api/system/status');
                systemStatus = await response.json();
                displayStatus();
            } catch (error) {
                alert('Error checking system: ' + error.message);
            }
        }

        function displayStatus() {
            const grid = document.getElementById('statusGrid');
            const continueBtn = document.getElementById('continueBtn');

            const items = [
                {
                    name: 'Avatar System',
                    status: systemStatus.avatar_system?.installed,
                    icon: '🎭'
                },
                {
                    name: 'T-FAN Cockpit',
                    status: systemStatus.tfan_cockpit?.installed,
                    icon: '🖥️'
                },
                {
                    name: 'Ollama (AI)',
                    status: systemStatus.ollama?.installed,
                    icon: '🤖'
                },
                {
                    name: 'Dependencies',
                    status: systemStatus.dependencies?.python && systemStatus.dependencies?.ffmpeg,
                    icon: '📦'
                }
            ];

            grid.innerHTML = items.map(item => `
                <div class="status-card ${item.status ? 'installed' : 'missing'}">
                    <h3>
                        <span class="status-icon ${item.status ? 'ok' : 'warn'}"></span>
                        ${item.icon} ${item.name}
                    </h3>
                    <p>${item.status ? 'Installed' : 'Not installed'}</p>
                </div>
            `).join('');

            continueBtn.disabled = false;
        }

        function goToStep(stepNum) {
            document.querySelectorAll('.step').forEach(s => s.classList.remove('active'));
            document.getElementById('step' + stepNum).classList.add('active');

            if (stepNum === 2) {
                showInstallOptions();
            }
        }

        function showInstallOptions() {
            const container = document.getElementById('installItems');
            const options = [];

            if (!systemStatus.dependencies?.ffmpeg || !systemStatus.ollama?.installed) {
                options.push({
                    title: 'System Dependencies',
                    desc: 'Install FFmpeg, Ollama, and other required packages',
                    action: 'installDeps'
                });
            }

            if (!systemStatus.tfan_cockpit?.installed) {
                options.push({
                    title: 'T-FAN Cockpit',
                    desc: 'Download the T-FAN spaceship interface from GitHub',
                    action: 'installTFAN'
                });
            }

            if (systemStatus.tfan_cockpit?.installed) {
                options.push({
                    title: 'Fix WebKit',
                    desc: 'Fix WebKit compatibility issues in T-FAN',
                    action: 'fixWebKit'
                });
            }

            if (options.length === 0) {
                container.innerHTML = '<p style="text-align: center; color: #28a745; font-size: 1.2rem;">✅ All components installed!</p>';
            } else {
                container.innerHTML = options.map(opt => `
                    <div class="install-item">
                        <h4>${opt.title}</h4>
                        <p>${opt.desc}</p>
                        <button class="btn" onclick="${opt.action}()" id="btn_${opt.action}">Install</button>
                        <div id="log_${opt.action}" class="log-output" style="display: none;"></div>
                    </div>
                `).join('');
            }
        }

        async function installDeps() {
            const btn = document.getElementById('btn_installDeps');
            const log = document.getElementById('log_installDeps');

            btn.disabled = true;
            btn.innerHTML = '<span class="loading"></span> Installing...';
            log.style.display = 'block';
            log.textContent = 'Starting dependency installation...\\n';

            try {
                const response = await fetch('/api/system/install-dependencies', { method: 'POST' });
                const result = await response.json();

                log.textContent += result.output;

                if (result.success) {
                    btn.innerHTML = '✅ Installed';
                    btn.className = 'btn btn-success';
                } else {
                    btn.innerHTML = '❌ Failed';
                    btn.disabled = false;
                }
            } catch (error) {
                log.textContent += '\\nError: ' + error.message;
                btn.innerHTML = '❌ Error';
                btn.disabled = false;
            }
        }

        async function installTFAN() {
            const btn = document.getElementById('btn_installTFAN');
            const log = document.getElementById('log_installTFAN');

            btn.disabled = true;
            btn.innerHTML = '<span class="loading"></span> Downloading...';
            log.style.display = 'block';
            log.textContent = 'Downloading T-FAN from GitHub...\\n';

            try {
                const response = await fetch('/api/system/install-tfan', { method: 'POST' });
                const result = await response.json();

                log.textContent += result.message + '\\n';

                if (result.success) {
                    btn.innerHTML = '✅ Downloaded';
                    btn.className = 'btn btn-success';
                } else {
                    btn.innerHTML = '⚠️ Manual Download Required';
                    if (result.manual_url) {
                        log.textContent += '\\nPlease download from: ' + result.manual_url;
                        log.textContent += '\\nExtract to: ' + result.install_path;
                    }
                }
            } catch (error) {
                log.textContent += '\\nError: ' + error.message;
                btn.innerHTML = '❌ Error';
                btn.disabled = false;
            }
        }

        async function fixWebKit() {
            const btn = document.getElementById('btn_fixWebKit');
            const log = document.getElementById('log_fixWebKit');

            btn.disabled = true;
            btn.innerHTML = '<span class="loading"></span> Fixing...';
            log.style.display = 'block';
            log.textContent = 'Fixing WebKit compatibility...\\n';

            try {
                const response = await fetch('/api/system/fix-webkit', { method: 'POST' });
                const result = await response.json();

                log.textContent += result.output;

                if (result.success) {
                    btn.innerHTML = '✅ Fixed';
                    btn.className = 'btn btn-success';
                } else {
                    btn.innerHTML = '❌ Failed';
                    log.textContent += '\\nError: ' + result.error;
                }
            } catch (error) {
                log.textContent += '\\nError: ' + error.message;
                btn.innerHTML = '❌ Error';
                btn.disabled = false;
            }
        }

        async function launch(component) {
            try {
                const response = await fetch(`/api/system/launch/${component}`, { method: 'POST' });
                const result = await response.json();

                if (result.success) {
                    alert(`${component} launched successfully!`);
                } else {
                    alert(`Failed to launch ${component}`);
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }

        function goToChat() {
            window.location.href = '/';
        }

        // Auto-check on load
        checkSystem();
    </script>
</body>
</html>
    """


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
        <h1>Multi-AI Workspace v0.2</h1>
        <div class="subtitle">Phase 2: Perspectives Mixer • Context Packs • Response Storage • Export</div>
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
