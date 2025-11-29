"""API routes for talking avatar."""

import asyncio
import uuid
from pathlib import Path
from typing import Dict
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import functools

from .models import GenerateRequest, GenerateResponse, HealthResponse, StatusResponse
from ..avatar_engine import AvatarGenerator
from ..config import settings
from .. import __version__

router = APIRouter()

# Global avatar generator instance
avatar_generator: AvatarGenerator | None = None

# Thread pool for blocking avatar generation operations
_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="avatar_gen")

# Job tracking
jobs: Dict[str, Dict] = {}


def get_generator() -> AvatarGenerator:
    """Get or create avatar generator instance."""
    global avatar_generator
    if avatar_generator is None:
        try:
            avatar_generator = AvatarGenerator(
                device=settings.device,
                output_fps=settings.output_fps,
                output_resolution=settings.output_resolution
            )
            # Load models if available
            model_path = settings.model_cache_dir / "wav2lip_model.pth"
            if model_path.exists():
                avatar_generator.load_models(model_path)
        except ImportError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Avatar generation unavailable: {str(e)}"
            )
    return avatar_generator


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=__version__,
        device=settings.device
    )


@router.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image for avatar generation.

    Args:
        file: Image file (PNG, JPG, JPEG)

    Returns:
        Filename of uploaded image
    """
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
    """Upload an audio file for avatar generation.

    Args:
        file: Audio file (WAV, MP3, etc.)

    Returns:
        Filename of uploaded audio
    """
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


async def process_avatar_generation(job_id: str, request: GenerateRequest):
    """Background task to process avatar generation.

    Args:
        job_id: Job ID
        request: Generation request
    """
    try:
        # Update job status
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["progress"] = 10

        # Get generator
        generator = get_generator()

        # Get file paths
        image_path = settings.upload_dir / request.image_filename
        audio_path = settings.upload_dir / request.audio_filename

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {request.image_filename}")
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {request.audio_filename}")

        jobs[job_id]["progress"] = 30

        # Generate avatar in thread pool to avoid blocking event loop
        output_path = settings.output_dir / f"{job_id}.mp4"
        loop = asyncio.get_event_loop()

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
            timeout=120.0  # 2 minute timeout for avatar generation
        )

        jobs[job_id]["progress"] = 90

        # Prepare response
        if result.success:
            response = GenerateResponse(
                success=True,
                video_url=f"/download/{job_id}.mp4",
                video_filename=f"{job_id}.mp4",
                duration=result.duration,
                frames_generated=result.frames_generated
            )
        else:
            response = GenerateResponse(
                success=False,
                error_message=result.error_message
            )

        # Update job
        jobs[job_id]["status"] = "completed" if result.success else "failed"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["result"] = response

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["progress"] = 0
        jobs[job_id]["result"] = GenerateResponse(
            success=False,
            error_message=str(e)
        )


@router.post("/generate", response_model=GenerateResponse)
async def generate_avatar(
    background_tasks: BackgroundTasks,
    request: GenerateRequest
):
    """Generate talking avatar video (synchronous).

    Args:
        background_tasks: Background tasks
        request: Generation request with uploaded filenames

    Returns:
        Generation result with video URL
    """
    # Get generator
    generator = get_generator()

    # Get file paths
    image_path = settings.upload_dir / request.image_filename
    audio_path = settings.upload_dir / request.audio_filename

    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image file not found: {request.image_filename}")
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {request.audio_filename}")

    # Generate avatar in thread pool to avoid blocking event loop
    job_id = uuid.uuid4().hex
    output_path = settings.output_dir / f"{job_id}.mp4"

    loop = asyncio.get_event_loop()
    try:
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
            timeout=120.0  # 2 minute timeout for avatar generation
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Avatar generation timed out after 2 minutes"
        )

    # Return response
    if result.success:
        return GenerateResponse(
            success=True,
            video_url=f"/download/{job_id}.mp4",
            video_filename=f"{job_id}.mp4",
            duration=result.duration,
            frames_generated=result.frames_generated
        )
    else:
        return GenerateResponse(
            success=False,
            error_message=result.error_message
        )


@router.post("/generate/async", response_model=StatusResponse)
async def generate_avatar_async(
    background_tasks: BackgroundTasks,
    request: GenerateRequest
):
    """Generate talking avatar video (asynchronous with job tracking).

    Args:
        background_tasks: Background tasks
        request: Generation request

    Returns:
        Job status with job ID for tracking
    """
    # Create job
    job_id = uuid.uuid4().hex
    jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "result": None
    }

    # Add background task
    background_tasks.add_task(process_avatar_generation, job_id, request)

    return StatusResponse(
        job_id=job_id,
        status="pending",
        progress=0
    )


@router.get("/status/{job_id}", response_model=StatusResponse)
async def get_job_status(job_id: str):
    """Get status of async generation job.

    Args:
        job_id: Job ID

    Returns:
        Job status and result
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    return StatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        result=job["result"]
    )


@router.get("/download/{filename}")
async def download_video(filename: str):
    """Download generated video.

    Args:
        filename: Video filename

    Returns:
        Video file
    """
    filepath = settings.output_dir / filename

    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    return FileResponse(
        path=filepath,
        media_type="video/mp4",
        filename=filename
    )


@router.delete("/cleanup/{filename}")
async def cleanup_file(filename: str):
    """Delete uploaded or generated file.

    Args:
        filename: File to delete

    Returns:
        Success message
    """
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
