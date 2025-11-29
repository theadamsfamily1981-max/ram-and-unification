"""Avatar configuration settings loader."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PerformanceSettings:
    """Performance configuration."""
    max_avatar_workers: int
    max_tts_workers: int
    gpu_enabled: bool
    device: str
    max_queue_size: int


@dataclass
class TimeoutSettings:
    """Timeout configuration."""
    avatar_generation: float
    tts_generation: float
    rvc_conversion: float
    ollama_timeout: float


@dataclass
class CacheSettings:
    """Cache configuration."""
    enabled: bool
    cache_dir: Path
    max_cache_size_mb: int
    cache_ttl_hours: int
    compress: bool


@dataclass
class VoiceSettings:
    """Voice/TTS configuration."""
    engine: str
    default_voice: str
    ara: Dict[str, Any]


@dataclass
class OobaboogaSettings:
    """Oobabooga integration settings."""
    enabled: bool
    api_url: str
    tts_extensions: list
    rvc_enabled: bool
    rvc_pitch: float
    rvc_index_rate: float
    streaming: bool


@dataclass
class AvatarGenSettings:
    """Avatar generation settings."""
    output_fps: int
    output_resolution: int
    quality_mode: str
    codec: str
    crf: int
    enable_streaming: bool


@dataclass
class MonitoringSettings:
    """Monitoring configuration."""
    enable_metrics: bool
    metrics_port: int
    detailed_logging: bool
    log_dir: Path


@dataclass
class AutoscalingSettings:
    """Auto-scaling configuration."""
    enabled: bool
    cpu_scale_up_threshold: int
    cpu_scale_down_threshold: int
    memory_threshold_gb: float
    min_workers: int
    max_workers: int


@dataclass
class RequestSettings:
    """Request management settings."""
    enable_priority_queue: bool
    enable_cancellation: bool
    max_concurrent_jobs: int
    job_retention_hours: int


@dataclass
class OutputSettings:
    """Output configuration."""
    output_dir: Path
    temp_dir: Path
    upload_dir: Path
    auto_cleanup: bool
    cleanup_after_days: int


class AvatarConfig:
    """Avatar API configuration manager."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration.

        Args:
            config_path: Path to config YAML file
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "avatar_config.yaml"

        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}

        # Load configuration
        self.load()

    def load(self):
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path) as f:
            self._config = yaml.safe_load(f)

        # Override with environment variables
        self._apply_env_overrides()

    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        # Performance
        if os.getenv("MAX_AVATAR_WORKERS"):
            self._config["performance"]["max_avatar_workers"] = int(os.getenv("MAX_AVATAR_WORKERS"))
        if os.getenv("MAX_TTS_WORKERS"):
            self._config["performance"]["max_tts_workers"] = int(os.getenv("MAX_TTS_WORKERS"))
        if os.getenv("GPU_ENABLED"):
            self._config["performance"]["gpu_enabled"] = os.getenv("GPU_ENABLED").lower() == "true"
        if os.getenv("DEVICE"):
            self._config["performance"]["device"] = os.getenv("DEVICE")

        # Timeouts
        if os.getenv("AVATAR_TIMEOUT"):
            self._config["timeouts"]["avatar_generation"] = float(os.getenv("AVATAR_TIMEOUT"))
        if os.getenv("TTS_TIMEOUT"):
            self._config["timeouts"]["tts_generation"] = float(os.getenv("TTS_TIMEOUT"))
        if os.getenv("RVC_TIMEOUT"):
            self._config["timeouts"]["rvc_conversion"] = float(os.getenv("RVC_TIMEOUT"))

        # Cache
        if os.getenv("CACHE_ENABLED"):
            self._config["cache"]["enabled"] = os.getenv("CACHE_ENABLED").lower() == "true"
        if os.getenv("CACHE_DIR"):
            self._config["cache"]["cache_dir"] = os.getenv("CACHE_DIR")

        # Voice
        if os.getenv("VOICE_ENGINE"):
            self._config["voice"]["engine"] = os.getenv("VOICE_ENGINE")
        if os.getenv("RVC_MODEL"):
            self._config["voice"]["ara"]["rvc_model"] = os.getenv("RVC_MODEL")

        # Oobabooga
        if os.getenv("OOBABOOGA_URL"):
            self._config["oobabooga"]["api_url"] = os.getenv("OOBABOOGA_URL")
        if os.getenv("OOBABOOGA_ENABLED"):
            self._config["oobabooga"]["enabled"] = os.getenv("OOBABOOGA_ENABLED").lower() == "true"

    @property
    def performance(self) -> PerformanceSettings:
        """Get performance settings."""
        perf = self._config["performance"]
        return PerformanceSettings(**perf)

    @property
    def timeouts(self) -> TimeoutSettings:
        """Get timeout settings."""
        return TimeoutSettings(**self._config["timeouts"])

    @property
    def cache(self) -> CacheSettings:
        """Get cache settings."""
        cache = self._config["cache"]
        cache["cache_dir"] = Path(cache["cache_dir"])
        return CacheSettings(**cache)

    @property
    def voice(self) -> VoiceSettings:
        """Get voice settings."""
        return VoiceSettings(**self._config["voice"])

    @property
    def oobabooga(self) -> OobaboogaSettings:
        """Get oobabooga settings."""
        return OobaboogaSettings(**self._config["oobabooga"])

    @property
    def avatar(self) -> AvatarGenSettings:
        """Get avatar generation settings."""
        return AvatarGenSettings(**self._config["avatar"])

    @property
    def monitoring(self) -> MonitoringSettings:
        """Get monitoring settings."""
        mon = self._config["monitoring"]
        mon["log_dir"] = Path(mon["log_dir"])
        return MonitoringSettings(**mon)

    @property
    def autoscaling(self) -> AutoscalingSettings:
        """Get autoscaling settings."""
        return AutoscalingSettings(**self._config["autoscaling"])

    @property
    def requests(self) -> RequestSettings:
        """Get request settings."""
        return RequestSettings(**self._config["requests"])

    @property
    def output(self) -> OutputSettings:
        """Get output settings."""
        out = self._config["output"]
        out["output_dir"] = Path(out["output_dir"])
        out["temp_dir"] = Path(out["temp_dir"])
        out["upload_dir"] = Path(out["upload_dir"])
        return OutputSettings(**out)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation, e.g., "performance.max_workers")
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value


# Global configuration instance
_config: Optional[AvatarConfig] = None


def get_config(config_path: Optional[Path] = None) -> AvatarConfig:
    """Get global configuration instance.

    Args:
        config_path: Optional config file path

    Returns:
        AvatarConfig instance
    """
    global _config

    if _config is None:
        _config = AvatarConfig(config_path)

    return _config


def reload_config():
    """Reload configuration from file."""
    global _config
    if _config is not None:
        _config.load()
