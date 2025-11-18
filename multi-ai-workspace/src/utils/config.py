"""Configuration loader for Multi-AI Workspace.

Loads and validates YAML configuration files for backends, routing rules,
and system settings.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from ..core.backend import AIProvider
from ..core.router import RoutingStrategy, RoutingRule
from ..integrations.claude_backend import ClaudeBackend
from ..integrations.nova_backend import NovaBackend
from ..integrations.pulse_backend import PulseBackend  # Ollama
from ..integrations.gemini_pulse_backend import GeminiPulseBackend  # Pulse (Gemini)
from ..integrations.grok_ara_backend import GrokAraBackend  # Ara (Grok)
from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class ConfigValidationError(Exception):
    """Configuration validation error."""
    message: str
    field: Optional[str] = None


class ConfigLoader:
    """
    YAML configuration loader for Multi-AI Workspace.

    Loads configuration from YAML files and creates AI backend instances
    and routing rules.
    """

    def __init__(self, config_path: str | Path):
        """
        Initialize configuration loader.

        Args:
            config_path: Path to YAML config file
        """
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load config
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

        logger.info(f"Configuration loaded from {config_path}")

    def load_backends(self) -> Dict[str, Any]:
        """
        Load and initialize AI backends from config.

        Returns:
            Dictionary of backend_name -> AIBackend instance

        Raises:
            ConfigValidationError: If configuration is invalid
        """
        backends_config = self.config.get("backends", {})

        if not backends_config:
            raise ConfigValidationError("No backends configured", "backends")

        backends = {}

        for name, config in backends_config.items():
            try:
                backend = self._create_backend(name, config)
                backends[name] = backend
                logger.info(f"Loaded backend: {name} ({backend.provider.value})")

            except Exception as e:
                logger.error(f"Failed to load backend '{name}': {e}")
                raise ConfigValidationError(f"Backend '{name}' configuration error: {e}", name)

        return backends

    def _create_backend(self, name: str, config: Dict[str, Any]):
        """
        Create backend instance from configuration.

        Args:
            name: Backend name
            config: Backend configuration

        Returns:
            AIBackend instance
        """
        provider = config.get("provider")
        enabled = config.get("enabled", True)

        if not enabled:
            logger.info(f"Backend '{name}' is disabled, skipping")
            return None

        if not provider:
            raise ConfigValidationError(f"No provider specified for backend '{name}'", f"backends.{name}.provider")

        # Create backend based on provider
        if provider == "claude" or provider == "anthropic":
            api_key = config.get("api_key") or self._get_env_var("ANTHROPIC_API_KEY")
            if not api_key:
                raise ConfigValidationError(
                    f"No API key for Claude backend '{name}'",
                    f"backends.{name}.api_key"
                )

            return ClaudeBackend(
                api_key=api_key,
                model=config.get("model", "sonnet"),
                name=name,
                config=config.get("config", {})
            )

        elif provider == "openai" or provider == "nova":
            api_key = config.get("api_key") or self._get_env_var("OPENAI_API_KEY")
            if not api_key:
                raise ConfigValidationError(
                    f"No API key for OpenAI backend '{name}'",
                    f"backends.{name}.api_key"
                )

            return NovaBackend(
                api_key=api_key,
                model=config.get("model", "gpt-4o"),
                name=name,
                config=config.get("config", {})
            )

        elif provider == "gemini" or provider == "pulse":
            # Pulse = Gemini (Google)
            api_key = config.get("api_key") or self._get_env_var("GOOGLE_API_KEY") or self._get_env_var("GEMINI_API_KEY")
            if not api_key:
                raise ConfigValidationError(
                    f"No API key for Gemini backend '{name}'",
                    f"backends.{name}.api_key"
                )

            return GeminiPulseBackend(
                api_key=api_key,
                model=config.get("model", "gemini-1.5-flash"),
                name=name,
                config=config.get("config", {})
            )

        elif provider == "grok" or provider == "ara":
            # Ara = Grok (X.AI) via Selenium
            username = config.get("username") or self._get_env_var("X_USERNAME")
            password = config.get("password") or self._get_env_var("X_PASSWORD")

            return GrokAraBackend(
                username=username,
                password=password,
                name=name,
                headless=config.get("headless", True),
                config=config.get("config", {})
            )

        elif provider == "ollama":
            # Legacy Ollama support
            return PulseBackend(
                model=config.get("model", "llama3.2"),
                name=name,
                base_url=config.get("base_url", "http://localhost:11434"),
                config=config.get("config", {})
            )

        else:
            raise ConfigValidationError(
                f"Unknown provider '{provider}' for backend '{name}'",
                f"backends.{name}.provider"
            )

    def load_routing_rules(self) -> List[RoutingRule]:
        """
        Load routing rules from config.

        Returns:
            List of RoutingRule objects
        """
        rules_config = self.config.get("routing", {}).get("rules", [])

        rules = []

        for rule_config in rules_config:
            try:
                rule = RoutingRule(
                    tags=rule_config.get("tags", []),
                    backends=rule_config.get("backends", []),
                    strategy=RoutingStrategy(rule_config.get("strategy", "single")),
                    priority=rule_config.get("priority", 0),
                    metadata=rule_config.get("metadata", {})
                )
                rules.append(rule)
                logger.debug(f"Loaded routing rule: {rule.tags} -> {rule.backends}")

            except Exception as e:
                logger.warning(f"Failed to load routing rule: {e}")

        logger.info(f"Loaded {len(rules)} routing rules")
        return rules

    def get_default_backend(self) -> Optional[str]:
        """
        Get default backend name from config.

        Returns:
            Default backend name or None
        """
        return self.config.get("routing", {}).get("default_backend")

    def get_system_config(self) -> Dict[str, Any]:
        """
        Get system configuration.

        Returns:
            System config dictionary
        """
        return self.config.get("system", {})

    def _get_env_var(self, var_name: str) -> Optional[str]:
        """
        Get environment variable.

        Args:
            var_name: Variable name

        Returns:
            Variable value or None
        """
        import os
        return os.environ.get(var_name)

    def validate(self) -> bool:
        """
        Validate configuration.

        Returns:
            True if valid

        Raises:
            ConfigValidationError: If invalid
        """
        # Check required sections
        if "backends" not in self.config:
            raise ConfigValidationError("Missing 'backends' section", "backends")

        # Validate backends
        backends_config = self.config.get("backends", {})
        if not backends_config:
            raise ConfigValidationError("No backends configured", "backends")

        # Validate routing
        routing_config = self.config.get("routing", {})
        if routing_config:
            # Check default backend exists
            default = routing_config.get("default_backend")
            if default and default not in backends_config:
                raise ConfigValidationError(
                    f"Default backend '{default}' not found in backends",
                    "routing.default_backend"
                )

            # Check rule backends exist
            rules = routing_config.get("rules", [])
            for i, rule in enumerate(rules):
                for backend in rule.get("backends", []):
                    if backend not in backends_config:
                        logger.warning(
                            f"Routing rule {i} references unknown backend '{backend}'"
                        )

        logger.info("Configuration validation passed")
        return True


def load_config(config_path: str | Path) -> ConfigLoader:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        ConfigLoader instance
    """
    loader = ConfigLoader(config_path)
    loader.validate()
    return loader
