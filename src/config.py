"""Configuration management for the Talking Avatar system."""

import os
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""

    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")

    # Model Configuration
    model_cache_dir: Path = Field(default=Path("./models"), description="Model cache directory")
    device: Literal["cuda", "cpu"] = Field(default="cpu", description="Device to use")
    batch_size: int = Field(default=1, description="Batch size for processing")

    # Processing Configuration
    max_video_length: int = Field(default=300, description="Maximum video length in seconds")
    output_format: str = Field(default="mp4", description="Output video format")
    output_fps: int = Field(default=25, description="Output video FPS")
    output_resolution: int = Field(default=512, description="Output resolution")

    # Storage
    upload_dir: Path = Field(default=Path("./uploads"), description="Upload directory")
    output_dir: Path = Field(default=Path("./outputs"), description="Output directory")
    temp_dir: Path = Field(default=Path("./temp"), description="Temporary directory")

    # API Security
    api_key: str | None = Field(default=None, description="API key for authentication")

    class Config:
        env_file = ".env"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
