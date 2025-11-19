"""T-FAN Client - Integration with Quanta-meis-nib-cis cockpit/topology system.

This client communicates with the T-FAN REST API to:
- Send voice commands to the cockpit
- Control topology visualization
- Manage workspace modes
- Get system metrics and status
- Control training jobs

T-FAN API lives in repo: theadamsfamily1981-max/Quanta-meis-nib-cis
"""

import asyncio
import httpx
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

from ..utils.logger import get_logger

logger = get_logger(__name__)


class TFANEndpoint(Enum):
    """T-FAN API endpoints."""
    COMMAND = "/api/ara/command"
    STATUS = "/api/ara/status"
    METRICS = "/api/metrics"
    TOPOLOGY = "/api/topology"
    TRAINING = "/api/training"
    WORKSPACE = "/api/workspace"


@dataclass
class TFANResponse:
    """Response from T-FAN API."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class TFANClient:
    """
    Client for communicating with T-FAN cockpit system.

    T-FAN (Quanta-meis-nib-cis) provides:
    - Cockpit HUD with metrics visualization
    - Topology visualization (network/system diagrams)
    - Workspace mode management
    - Training job control
    - System status reporting
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 30.0,
        api_key: Optional[str] = None
    ):
        """
        Initialize T-FAN client.

        Args:
            base_url: T-FAN API base URL
            timeout: Request timeout in seconds
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.api_key = api_key

        # HTTP client
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            headers=headers
        )

        self._connected = False
        logger.info(f"TFANClient initialized: {base_url}")

    async def check_connection(self) -> bool:
        """
        Check if T-FAN is reachable.

        Returns:
            True if connected
        """
        try:
            response = await self.client.get("/api/health")
            self._connected = response.status_code == 200
            return self._connected
        except Exception as e:
            logger.warning(f"T-FAN connection check failed: {e}")
            self._connected = False
            return False

    @property
    def is_connected(self) -> bool:
        """Return connection status."""
        return self._connected

    async def send_command(self, command: str) -> TFANResponse:
        """
        Send a voice command to T-FAN.

        This is the primary endpoint for voice macro execution.
        T-FAN interprets the command and executes appropriate actions.

        Args:
            command: Natural language command string

        Returns:
            TFANResponse with result
        """
        try:
            response = await self.client.post(
                TFANEndpoint.COMMAND.value,
                json={"command": command}
            )

            if response.status_code == 200:
                data = response.json()
                return TFANResponse(
                    success=True,
                    message=data.get("message", "Command executed"),
                    data=data.get("data")
                )
            else:
                return TFANResponse(
                    success=False,
                    message="Command failed",
                    error=f"HTTP {response.status_code}: {response.text}"
                )

        except httpx.ConnectError:
            logger.error(f"Cannot connect to T-FAN at {self.base_url}")
            return TFANResponse(
                success=False,
                message="T-FAN not reachable",
                error=f"Connection failed to {self.base_url}"
            )
        except Exception as e:
            logger.error(f"Error sending command to T-FAN: {e}")
            return TFANResponse(
                success=False,
                message="Command error",
                error=str(e)
            )

    async def get_status(self) -> TFANResponse:
        """
        Get system status from T-FAN.

        Returns:
            TFANResponse with status data
        """
        try:
            response = await self.client.get(TFANEndpoint.STATUS.value)

            if response.status_code == 200:
                data = response.json()
                return TFANResponse(
                    success=True,
                    message="Status retrieved",
                    data=data
                )
            else:
                return TFANResponse(
                    success=False,
                    message="Failed to get status",
                    error=f"HTTP {response.status_code}"
                )

        except Exception as e:
            logger.error(f"Error getting T-FAN status: {e}")
            return TFANResponse(
                success=False,
                message="Status error",
                error=str(e)
            )

    async def get_metrics(self, metric_type: str = "all") -> TFANResponse:
        """
        Get system metrics from T-FAN.

        Args:
            metric_type: Type of metrics (gpu, cpu, network, storage, all)

        Returns:
            TFANResponse with metrics data
        """
        try:
            response = await self.client.get(
                TFANEndpoint.METRICS.value,
                params={"type": metric_type}
            )

            if response.status_code == 200:
                data = response.json()
                return TFANResponse(
                    success=True,
                    message=f"{metric_type} metrics retrieved",
                    data=data
                )
            else:
                return TFANResponse(
                    success=False,
                    message="Failed to get metrics",
                    error=f"HTTP {response.status_code}"
                )

        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return TFANResponse(
                success=False,
                message="Metrics error",
                error=str(e)
            )

    async def control_topology(self, action: str, params: Optional[Dict] = None) -> TFANResponse:
        """
        Control topology visualization.

        Args:
            action: Action to perform (show, hide, fullscreen, mode)
            params: Additional parameters

        Returns:
            TFANResponse with result
        """
        try:
            payload = {"action": action}
            if params:
                payload["params"] = params

            response = await self.client.post(
                TFANEndpoint.TOPOLOGY.value,
                json=payload
            )

            if response.status_code == 200:
                data = response.json()
                return TFANResponse(
                    success=True,
                    message=f"Topology {action} executed",
                    data=data
                )
            else:
                return TFANResponse(
                    success=False,
                    message=f"Topology {action} failed",
                    error=f"HTTP {response.status_code}"
                )

        except Exception as e:
            logger.error(f"Error controlling topology: {e}")
            return TFANResponse(
                success=False,
                message="Topology error",
                error=str(e)
            )

    async def control_training(self, action: str, config: Optional[Dict] = None) -> TFANResponse:
        """
        Control training jobs.

        Args:
            action: Action (start, stop, pause, resume, status)
            config: Training configuration (for start action)

        Returns:
            TFANResponse with result
        """
        try:
            payload = {"action": action}
            if config:
                payload["config"] = config

            response = await self.client.post(
                TFANEndpoint.TRAINING.value,
                json=payload
            )

            if response.status_code == 200:
                data = response.json()
                return TFANResponse(
                    success=True,
                    message=f"Training {action} executed",
                    data=data
                )
            else:
                return TFANResponse(
                    success=False,
                    message=f"Training {action} failed",
                    error=f"HTTP {response.status_code}"
                )

        except Exception as e:
            logger.error(f"Error controlling training: {e}")
            return TFANResponse(
                success=False,
                message="Training error",
                error=str(e)
            )

    async def set_workspace_mode(self, mode: str) -> TFANResponse:
        """
        Set workspace mode (work, relax, focus).

        Args:
            mode: Workspace mode name

        Returns:
            TFANResponse with result
        """
        try:
            response = await self.client.post(
                TFANEndpoint.WORKSPACE.value,
                json={"mode": mode}
            )

            if response.status_code == 200:
                data = response.json()
                return TFANResponse(
                    success=True,
                    message=f"Workspace set to {mode} mode",
                    data=data
                )
            else:
                return TFANResponse(
                    success=False,
                    message=f"Failed to set {mode} mode",
                    error=f"HTTP {response.status_code}"
                )

        except Exception as e:
            logger.error(f"Error setting workspace mode: {e}")
            return TFANResponse(
                success=False,
                message="Workspace error",
                error=str(e)
            )

    async def cleanup(self):
        """Clean up client resources."""
        await self.client.aclose()
        logger.info("TFANClient cleaned up")

    def get_info(self) -> Dict[str, Any]:
        """
        Get client information.

        Returns:
            Dictionary with client details
        """
        return {
            "base_url": self.base_url,
            "connected": self._connected,
            "timeout": self.timeout,
            "endpoints": {
                "command": TFANEndpoint.COMMAND.value,
                "status": TFANEndpoint.STATUS.value,
                "metrics": TFANEndpoint.METRICS.value,
                "topology": TFANEndpoint.TOPOLOGY.value,
                "training": TFANEndpoint.TRAINING.value,
                "workspace": TFANEndpoint.WORKSPACE.value
            }
        }


# Convenience function for one-off commands
async def send_tfan_command(
    command: str,
    base_url: str = "http://localhost:8080"
) -> TFANResponse:
    """
    Send a one-off command to T-FAN.

    Args:
        command: Command string
        base_url: T-FAN API URL

    Returns:
        TFANResponse
    """
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        try:
            response = await client.post(
                TFANEndpoint.COMMAND.value,
                json={"command": command}
            )

            if response.status_code == 200:
                data = response.json()
                return TFANResponse(
                    success=True,
                    message=data.get("message", "Command executed"),
                    data=data.get("data")
                )
            else:
                return TFANResponse(
                    success=False,
                    message="Command failed",
                    error=f"HTTP {response.status_code}"
                )
        except Exception as e:
            return TFANResponse(
                success=False,
                message="Command error",
                error=str(e)
            )
