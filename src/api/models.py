"""API request and response models."""

from pydantic import BaseModel, Field
from typing import Optional


class GenerateRequest(BaseModel):
    """Request model for avatar generation."""

    image_filename: str = Field(..., description="Uploaded image filename")
    audio_filename: str = Field(..., description="Uploaded audio filename")
    output_fps: Optional[int] = Field(default=25, description="Output video FPS", ge=10, le=60)
    output_resolution: Optional[int] = Field(
        default=512,
        description="Output resolution",
        ge=256,
        le=1024
    )


class GenerateResponse(BaseModel):
    """Response model for avatar generation."""

    success: bool = Field(..., description="Whether generation was successful")
    video_url: Optional[str] = Field(None, description="URL to download generated video")
    video_filename: Optional[str] = Field(None, description="Generated video filename")
    duration: Optional[float] = Field(None, description="Video duration in seconds")
    frames_generated: Optional[int] = Field(None, description="Number of frames generated")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    device: str = Field(..., description="Device being used")


class StatusResponse(BaseModel):
    """Status response for job tracking."""

    job_id: str = Field(..., description="Job ID")
    status: str = Field(..., description="Job status (pending, processing, completed, failed)")
    progress: Optional[float] = Field(None, description="Progress percentage (0-100)")
    result: Optional[GenerateResponse] = Field(None, description="Result if completed")
