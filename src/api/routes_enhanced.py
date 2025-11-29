"""Enhanced API routes for talking avatar with all optimizations.

Features:
- Configurable timeouts and workers
- Avatar generation caching
- GPU auto-detection
- WebSocket progress streaming
- Job cancellation support
- Detailed health checks
- Request queue with priority
"""

import asyncio
import uuid
import time
from pathlib import Path
from typing import Dict, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import functools
import logging

from .models import GenerateRequest, GenerateResponse, HealthResponse, StatusResponse
from ..avatar_engine import AvatarGenerator
from ..config import settings
from .. import __version__

# Import new modules
try:
    from src.config.avatar_settings import get_config
    from src.cache.avatar_cache import AvatarCache
    from src.utils.device_utils import get_optimal_device, get_device_info, check_gpu_requirements
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter()

# Global instances
avatar_generator: Optional[AvatarGenerator] = None
avatar_cache: Optional[AvatarCache] = None
_executor: Optional[ThreadPoolExecutor] = None
_config = None

# Job tracking with cancellation support
jobs: Dict[str, Dict] = {}
job_futures: Dict[str, asyncio.Future] = {}

# WebSocket connections for progress streaming
websocket_connections: Dict[str, WebSocket] = {}


def initialize_globals():
    """Initialize global instances with configuration."""
    global avatar_generator, avatar_cache, _executor, _config

    # Load configuration
    if CONFIG_AVAILABLE:
        try:
            _config = get_config()
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
            _config = None

    # Initialize cache
    if _config and _config.cache.enabled:
        avatar_cache = AvatarCache(
            cache_dir=_config.cache.cache_dir,
            max_size_mb=_config.cache.max_cache_size_mb,
            ttl_hours=_config.cache.cache_ttl_hours,
            compress=_config.cache.compress
        )
        logger.info("Avatar cache initialized")

    # Initialize thread pool with config
    max_workers = _config.performance.max_avatar_workers if _config else 3
    _executor = ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="avatar_gen"
    )
    logger.info(f"Thread pool initialized with {max_workers} workers")


def get_generator() -> AvatarGenerator:
    """Get or create avatar generator instance with auto device selection."""
    global avatar_generator

    if avatar_generator is None:
        # Auto-detect device
        if CONFIG_AVAILABLE:
            device = get_optimal_device()
        else:
            device = settings.device

        try:
            avatar_generator = AvatarGenerator(
                device=device,
                output_fps=_config.avatar.output_fps if _config else settings.output_fps,
                output_resolution=_config.avatar.output_resolution if _config else settings.output_resolution
            )

            # Load models if available
            model_path = settings.model_cache_dir / "wav2lip_model.pth"
            if model_path.exists():
                avatar_generator.load_models(model_path)

            logger.info(f"Avatar generator initialized (device: {device})")

        except ImportError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Avatar generation unavailable: {str(e)}"
            )

    return avatar_generator


@router.on_event("startup")
async def startup():
    """Initialize on startup."""
    initialize_globals()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint."""
    device = get_optimal_device() if CONFIG_AVAILABLE else settings.device
    return HealthResponse(
        status="healthy",
        version=__version__,
        device=device
    )


@router.get("/health/detailed")
async def detailed_health_check():
    """Comprehensive health check with system info."""
    if not CONFIG_AVAILABLE:
        return JSONResponse({
            "status": "limited",
            "message": "Enhanced features not available"
        })

    device_info = get_device_info()
    gpu_check = check_gpu_requirements()

    # Cache stats
    cache_stats = avatar_cache.get_stats() if avatar_cache else None

    # Thread pool status
    executor_status = {
        "max_workers": _executor._max_workers if _executor else 0,
        "active_threads": len(_executor._threads) if _executor else 0
    }

    return JSONResponse({
        "status": "healthy",
        "version": __version__,
        "device": device_info,
        "gpu_requirements": gpu_check,
        "cache": cache_stats,
        "thread_pool": executor_status,
        "active_jobs": len([j for j in jobs.values() if j["status"] == "processing"]),
        "total_jobs": len(jobs)
    })


@router.get("/config")
async def get_configuration():
    """Get current configuration."""
    if not _config:
        raise HTTPException(status_code=503, detail="Configuration not available")

    return JSONResponse({
        "performance": {
            "max_avatar_workers": _config.performance.max_avatar_workers,
            "max_tts_workers": _config.performance.max_tts_workers,
            "gpu_enabled": _config.performance.gpu_enabled,
            "device": _config.performance.device
        },
        "timeouts": {
            "avatar_generation": _config.timeouts.avatar_generation,
            "tts_generation": _config.timeouts.tts_generation
        },
        "cache": {
            "enabled": _config.cache.enabled,
            "max_size_mb": _config.cache.max_cache_size_mb,
            "ttl_hours": _config.cache.cache_ttl_hours
        }
    })


@router.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image for avatar generation."""
    # Validate file type
    allowed_types = ["image/png", "image/jpeg", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
        )

    # Generate unique filename
    ext = Path(file.filename).suffix
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = settings.upload_dir / filename

    # Save file
    async with aiofiles.open(filepath, 'wb') as f:
        content = await file.read()
        await f.write(content)

    return {"filename": filename, "message": "Image uploaded successfully"}


@router.post("/upload/audio")
async def upload_audio(file: UploadFile = File(...)):
    """Upload an audio file for avatar generation."""
    # Validate file type
    allowed_types = ["audio/wav", "audio/mpeg", "audio/mp3", "audio/x-wav"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: WAV, MP3"
        )

    # Generate unique filename
    ext = Path(file.filename).suffix
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = settings.upload_dir / filename

    # Save file
    async with aiofiles.open(filepath, 'wb') as f:
        content = await file.read()
        await f.write(content)

    return {"filename": filename, "message": "Audio uploaded successfully"}


async def send_progress_update(job_id: str, progress: int, status: str, message: str = ""):
    """Send progress update via WebSocket if connected."""
    if job_id in websocket_connections:
        try:
            ws = websocket_connections[job_id]
            await ws.send_json({
                "job_id": job_id,
                "progress": progress,
                "status": status,
                "message": message
            })
        except:
            # Connection closed
            if job_id in websocket_connections:
                del websocket_connections[job_id]


async def process_avatar_generation(job_id: str, request: GenerateRequest):
    """Background task to process avatar generation with caching and progress updates."""
    try:
        # Update job status
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["progress"] = 10
        await send_progress_update(job_id, 10, "processing", "Initializing...")

        # Get generator
        generator = get_generator()

        # Get file paths
        image_path = settings.upload_dir / request.image_filename
        audio_path = settings.upload_dir / request.audio_filename

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {request.image_filename}")
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {request.audio_filename}")

        jobs[job_id]["progress"] = 20
        await send_progress_update(job_id, 20, "processing", "Checking cache...")

        # Check cache
        cache_hit = False
        if avatar_cache:
            cache_key = avatar_cache.generate_key(image_path, audio_path)
            cached_video = avatar_cache.get(cache_key)

            if cached_video:
                logger.info(f"Cache hit for job {job_id}")
                cache_hit = True
                output_path = settings.output_dir / f"{job_id}.mp4"
                import shutil
                shutil.copy(cached_video, output_path)

                # Create success result
                result_success = True
                result_duration = 0  # Would need to read from cache metadata
                result_frames = 0

        if not cache_hit:
            jobs[job_id]["progress"] = 30
            await send_progress_update(job_id, 30, "processing", "Generating avatar...")

            # Generate avatar in thread pool to avoid blocking event loop
            output_path = settings.output_dir / f"{job_id}.mp4"
            loop = asyncio.get_event_loop()

            # Get timeout from config
            timeout = _config.timeouts.avatar_generation if _config else 120.0

            result = await asyncio.wait_for(
                loop.run_in_executor(
                    _executor,
                    functools.partial(
                        generator.generate,
                        image_input=image_path,
                        audio_input=audio_path,
                        output_path=output_path,
                        temp_dir=settings.temp_dir
                    )
                ),
                timeout=timeout
            )

            result_success = result.success
            result_duration = result.duration
            result_frames = result.frames_generated

            # Cache the result if successful
            if result_success and avatar_cache:
                avatar_cache.put(cache_key, output_path, {
                    "duration": result_duration,
                    "frames": result_frames,
                    "job_id": job_id
                })

        jobs[job_id]["progress"] = 90
        await send_progress_update(job_id, 90, "processing", "Finalizing...")

        # Prepare response
        if result_success or cache_hit:
            response = GenerateResponse(
                success=True,
                video_url=f"/download/{job_id}.mp4",
                video_filename=f"{job_id}.mp4",
                duration=result_duration if not cache_hit else 0,
                frames_generated=result_frames if not cache_hit else 0
            )
        else:
            response = GenerateResponse(
                success=False,
                error_message=result.error_message if hasattr(result, 'error_message') else "Unknown error"
            )

        # Update job
        jobs[job_id]["status"] = "completed" if (result_success or cache_hit) else "failed"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["result"] = response
        await send_progress_update(job_id, 100, "completed", "Done!")

    except asyncio.CancelledError:
        jobs[job_id]["status"] = "cancelled"
        jobs[job_id]["progress"] = 0
        await send_progress_update(job_id, 0, "cancelled", "Job cancelled")
        logger.info(f"Job {job_id} cancelled")

    except asyncio.TimeoutError:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["progress"] = 0
        jobs[job_id]["result"] = GenerateResponse(
            success=False,
            error_message=f"Generation timed out after {timeout}s"
        )
        await send_progress_update(job_id, 0, "failed", "Timeout")
        logger.error(f"Job {job_id} timed out")

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["progress"] = 0
        jobs[job_id]["result"] = GenerateResponse(
            success=False,
            error_message=str(e)
        )
        await send_progress_update(job_id, 0, "failed", f"Error: {str(e)}")
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)


@router.post("/generate/async", response_model=StatusResponse)
async def generate_avatar_async(
    background_tasks: BackgroundTasks,
    request: GenerateRequest
):
    """Generate talking avatar video (asynchronous with job tracking and caching)."""
    # Create job
    job_id = uuid.uuid4().hex
    jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "result": None,
        "created_at": time.time()
    }

    # Add background task
    task = asyncio.create_task(process_avatar_generation(job_id, request))
    job_futures[job_id] = task

    logger.info(f"Created job {job_id}")

    return StatusResponse(
        job_id=job_id,
        status="pending",
        progress=0
    )


@router.get("/status/{job_id}", response_model=StatusResponse)
async def get_job_status(job_id: str):
    """Get status of async generation job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    return StatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        result=job["result"]
    )


@router.delete("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a running avatar generation job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    if job_id in job_futures:
        job_futures[job_id].cancel()
        logger.info(f"Cancelled job {job_id}")
        return {"message": f"Job {job_id} cancelled"}

    return {"message": f"Job {job_id} cannot be cancelled (already completed or not found)"}


@router.websocket("/ws/progress/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time progress updates."""
    await websocket.accept()
    websocket_connections[job_id] = websocket

    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(1)

            # Check if job is complete
            if job_id in jobs and jobs[job_id]["status"] in ["completed", "failed", "cancelled"]:
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for job {job_id}")
    finally:
        if job_id in websocket_connections:
            del websocket_connections[job_id]


@router.get("/download/{filename}")
async def download_video(filename: str):
    """Download generated video."""
    filepath = settings.output_dir / filename

    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    return FileResponse(
        path=filepath,
        media_type="video/mp4",
        filename=filename
    )


@router.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics."""
    if not avatar_cache:
        raise HTTPException(status_code=503, detail="Cache not enabled")

    return JSONResponse(avatar_cache.get_stats())


@router.delete("/cache/clear")
async def clear_cache():
    """Clear entire avatar cache."""
    if not avatar_cache:
        raise HTTPException(status_code=503, detail="Cache not enabled")

    avatar_cache.clear()
    return {"message": "Cache cleared successfully"}


@router.post("/cache/cleanup")
async def cleanup_cache():
    """Cleanup expired cache entries."""
    if not avatar_cache:
        raise HTTPException(status_code=503, detail="Cache not enabled")

    avatar_cache.cleanup_expired()
    return {"message": "Cache cleanup completed"}


@router.delete("/cleanup/{filename}")
async def cleanup_file(filename: str):
    """Delete uploaded or generated file."""
    # Check in uploads
    upload_path = settings.upload_dir / filename
    output_path = settings.output_dir / filename

    deleted = False
    if upload_path.exists():
        upload_path.unlink()
        deleted = True
    if output_path.exists():
        output_path.unlink()
        deleted = True

    if not deleted:
        raise HTTPException(status_code=404, detail="File not found")

    return {"message": f"File {filename} deleted successfully"}
