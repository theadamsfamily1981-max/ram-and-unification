"""Voice Macros - Voice command processing and T-FAN integration.

Routes voice commands from ASR to appropriate handlers:
- tfan_command: Send to T-FAN cockpit API
- ara_mode: Internal avatar mode change
- ara_avatar: Visual avatar configuration
- ara_action: Internal Ara actions (Colab, macros, etc.)
"""

import yaml
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

from ..integrations.tfan_client import TFANClient, TFANResponse
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MacroType(Enum):
    """Types of voice macros."""
    TFAN_COMMAND = "tfan_command"
    ARA_MODE = "ara_mode"
    ARA_AVATAR = "ara_avatar"
    ARA_ACTION = "ara_action"
    SHELL = "shell"


@dataclass
class MacroMatch:
    """Result of matching voice input to a macro."""
    matched: bool
    macro_name: Optional[str] = None
    macro_type: Optional[MacroType] = None
    command: Optional[Any] = None
    description: Optional[str] = None
    speak_summary: Optional[str] = None
    confidence: float = 0.0


@dataclass
class MacroResult:
    """Result of executing a macro."""
    success: bool
    message: str
    spoken_response: str
    data: Optional[Dict[str, Any]] = None


class VoiceMacroProcessor:
    """
    Voice Macro Processor.

    Loads voice macros from YAML configuration and routes
    voice commands to appropriate handlers.
    """

    def __init__(
        self,
        config_path: str = "config/voice_macros.yaml",
        tfan_client: Optional[TFANClient] = None,
        tfan_base_url: str = "http://localhost:8080"
    ):
        """
        Initialize Voice Macro Processor.

        Args:
            config_path: Path to voice_macros.yaml
            tfan_client: Optional pre-configured T-FAN client
            tfan_base_url: T-FAN API base URL (if client not provided)
        """
        self.config_path = Path(config_path)
        self.macros: Dict[str, Dict[str, Any]] = {}

        # T-FAN client
        self.tfan_client = tfan_client or TFANClient(base_url=tfan_base_url)

        # Internal state
        self._current_mode = "default"
        self._current_avatar = {
            "profile": "default",
            "style": "neutral",
            "mood": "neutral"
        }

        # Load macros
        self._load_macros()

        logger.info(f"VoiceMacroProcessor initialized with {len(self.macros)} macros")

    def _load_macros(self):
        """Load macros from YAML configuration."""
        if not self.config_path.exists():
            logger.warning(f"Macro config not found: {self.config_path}")
            return

        try:
            with open(self.config_path) as f:
                config = yaml.safe_load(f)

            self.macros = config.get("macros", {})
            logger.info(f"Loaded {len(self.macros)} macros from {self.config_path}")

        except Exception as e:
            logger.error(f"Error loading macros: {e}")
            self.macros = {}

    def reload_macros(self):
        """Reload macros from configuration file."""
        self._load_macros()
        return len(self.macros)

    def match_macro(self, voice_input: str) -> MacroMatch:
        """
        Match voice input to a macro.

        Args:
            voice_input: Normalized voice input text

        Returns:
            MacroMatch with result
        """
        # Normalize input
        normalized = voice_input.lower().strip()

        # Direct match
        if normalized in self.macros:
            macro = self.macros[normalized]
            return MacroMatch(
                matched=True,
                macro_name=normalized,
                macro_type=MacroType(macro.get("type", "ara_action")),
                command=macro.get("command"),
                description=macro.get("description"),
                speak_summary=macro.get("speak_summary"),
                confidence=1.0
            )

        # Fuzzy match - check for partial matches
        best_match = None
        best_score = 0.0

        for macro_name, macro in self.macros.items():
            score = self._fuzzy_match_score(normalized, macro_name)
            if score > best_score and score > 0.7:  # 70% threshold
                best_score = score
                best_match = macro_name

        if best_match:
            macro = self.macros[best_match]
            return MacroMatch(
                matched=True,
                macro_name=best_match,
                macro_type=MacroType(macro.get("type", "ara_action")),
                command=macro.get("command"),
                description=macro.get("description"),
                speak_summary=macro.get("speak_summary"),
                confidence=best_score
            )

        return MacroMatch(matched=False)

    def _fuzzy_match_score(self, input_text: str, macro_name: str) -> float:
        """
        Calculate fuzzy match score between input and macro name.

        Args:
            input_text: User input
            macro_name: Macro trigger phrase

        Returns:
            Score from 0.0 to 1.0
        """
        # Simple word overlap scoring
        input_words = set(input_text.split())
        macro_words = set(macro_name.split())

        if not macro_words:
            return 0.0

        overlap = len(input_words & macro_words)
        score = overlap / len(macro_words)

        # Bonus for exact substring match
        if macro_name in input_text or input_text in macro_name:
            score = min(1.0, score + 0.3)

        return score

    async def execute_macro(self, match: MacroMatch) -> MacroResult:
        """
        Execute a matched macro.

        Args:
            match: MacroMatch result from match_macro

        Returns:
            MacroResult with execution result
        """
        if not match.matched:
            return MacroResult(
                success=False,
                message="No macro matched",
                spoken_response="I didn't recognize that command."
            )

        logger.info(f"Executing macro: {match.macro_name} (type: {match.macro_type})")

        # Route to appropriate handler
        if match.macro_type == MacroType.TFAN_COMMAND:
            return await self._execute_tfan_command(match)
        elif match.macro_type == MacroType.ARA_MODE:
            return await self._execute_ara_mode(match)
        elif match.macro_type == MacroType.ARA_AVATAR:
            return await self._execute_ara_avatar(match)
        elif match.macro_type == MacroType.ARA_ACTION:
            return await self._execute_ara_action(match)
        elif match.macro_type == MacroType.SHELL:
            return await self._execute_shell(match)
        else:
            return MacroResult(
                success=False,
                message=f"Unknown macro type: {match.macro_type}",
                spoken_response="I don't know how to handle that type of command."
            )

    async def _execute_tfan_command(self, match: MacroMatch) -> MacroResult:
        """Execute T-FAN command macro."""
        command = match.command
        if not command:
            return MacroResult(
                success=False,
                message="No command specified",
                spoken_response="This macro doesn't have a command configured."
            )

        # Send to T-FAN
        response = await self.tfan_client.send_command(command)

        if response.success:
            return MacroResult(
                success=True,
                message=response.message,
                spoken_response=f"Done. {response.message}",
                data=response.data
            )
        else:
            return MacroResult(
                success=False,
                message=response.error or "T-FAN command failed",
                spoken_response=f"I couldn't complete that command. {response.error}"
            )

    async def _execute_ara_mode(self, match: MacroMatch) -> MacroResult:
        """Execute Ara mode change."""
        mode = match.command
        if not mode:
            return MacroResult(
                success=False,
                message="No mode specified",
                spoken_response="This macro doesn't specify a mode."
            )

        # Update internal mode
        self._current_mode = mode

        # Determine response based on mode
        mode_responses = {
            "focus": "Switching to focus mode. I'll keep things concise and task-oriented.",
            "chill": "Switching to chill mode. Let's take it easy.",
            "default": "Switching back to default mode.",
            "professional": "Switching to professional mode. All business.",
            "creative": "Switching to creative mode. Let's brainstorm!"
        }

        spoken = mode_responses.get(
            mode,
            f"Switching to {mode} mode."
        )

        return MacroResult(
            success=True,
            message=f"Mode changed to {mode}",
            spoken_response=spoken,
            data={"mode": mode}
        )

    async def _execute_ara_avatar(self, match: MacroMatch) -> MacroResult:
        """Execute Ara avatar configuration change."""
        config = match.command
        if not config:
            return MacroResult(
                success=False,
                message="No avatar config specified",
                spoken_response="This macro doesn't have avatar settings."
            )

        # Update avatar state
        if isinstance(config, dict):
            self._current_avatar.update(config)
        else:
            self._current_avatar["profile"] = str(config)

        profile = self._current_avatar.get("profile", "default")
        mood = self._current_avatar.get("mood", "neutral")

        return MacroResult(
            success=True,
            message=f"Avatar updated: {profile} / {mood}",
            spoken_response=f"Switching to {profile} look with {mood} mood.",
            data={"avatar": self._current_avatar}
        )

    async def _execute_ara_action(self, match: MacroMatch) -> MacroResult:
        """Execute internal Ara action."""
        action = match.command
        if not action:
            return MacroResult(
                success=False,
                message="No action specified",
                spoken_response="This macro doesn't have an action."
            )

        # Handle specific actions
        if action == "list_macros":
            return await self._action_list_macros()
        elif action == "explain_macro":
            return await self._action_explain_macro()
        elif action == "explain_capabilities":
            return await self._action_explain_capabilities()
        elif action == "macro_create_wizard":
            return await self._action_create_wizard()
        elif action == "macro_delete_wizard":
            return await self._action_delete_wizard()
        elif action == "offload_notebook_to_colab":
            return await self._action_colab_offload()
        elif action == "sync_notebooks_with_drive":
            return await self._action_sync_notebooks()
        else:
            return MacroResult(
                success=False,
                message=f"Unknown action: {action}",
                spoken_response=f"I don't know how to perform the action: {action}"
            )

    async def _execute_shell(self, match: MacroMatch) -> MacroResult:
        """Execute shell command (disabled by default for safety)."""
        return MacroResult(
            success=False,
            message="Shell commands disabled",
            spoken_response="Shell commands are disabled for safety."
        )

    # =========================================================================
    # Internal Actions
    # =========================================================================

    async def _action_list_macros(self) -> MacroResult:
        """List all available macros."""
        if not self.macros:
            return MacroResult(
                success=True,
                message="No macros configured",
                spoken_response="You don't have any macros configured yet."
            )

        # Group by type
        groups: Dict[str, List[str]] = {}
        for name, macro in self.macros.items():
            macro_type = macro.get("type", "unknown")
            if macro_type not in groups:
                groups[macro_type] = []
            groups[macro_type].append(name)

        # Build response
        lines = ["Here are your voice macros:"]
        for group_type, names in groups.items():
            lines.append(f"\n{group_type.replace('_', ' ').title()}:")
            for name in sorted(names)[:5]:  # Limit to 5 per group for speech
                lines.append(f"  - {name}")
            if len(names) > 5:
                lines.append(f"  ... and {len(names) - 5} more")

        spoken = f"You have {len(self.macros)} macros. " + \
                 f"Some examples: {', '.join(list(self.macros.keys())[:5])}. " + \
                 "Say 'explain macro' to learn about a specific one."

        return MacroResult(
            success=True,
            message="\n".join(lines),
            spoken_response=spoken,
            data={"macros": list(self.macros.keys()), "groups": groups}
        )

    async def _action_explain_macro(self) -> MacroResult:
        """Start macro explanation flow."""
        return MacroResult(
            success=True,
            message="Ready to explain macro",
            spoken_response="Which macro would you like me to explain? Say the trigger phrase.",
            data={"awaiting": "macro_name"}
        )

    async def _action_explain_capabilities(self) -> MacroResult:
        """Explain Ara's capabilities."""
        capabilities = """
I can help you with:

**Cockpit Control:**
- Show metrics (GPU, CPU, network, storage)
- Control topology visualization
- Set workspace modes (work, relax, focus)

**Training:**
- Start and stop training jobs
- Check training status
- Run validation

**Voice Macros:**
- Execute custom voice commands
- Create and manage macros

**Research (via Offline Avatar):**
- Multi-AI research orchestration
- Literature reviews
- Comparative analysis

**Colab Integration:**
- Offload notebooks to Colab
- Sync with Google Drive

Say 'what macros do I have' to see your custom commands.
"""

        spoken = "I can control the cockpit metrics and topology, manage training jobs, " + \
                 "run voice macros, conduct multi-AI research, and offload work to Colab. " + \
                 "Ask me about specific capabilities or say 'what macros do I have' to see your commands."

        return MacroResult(
            success=True,
            message=capabilities,
            spoken_response=spoken,
            data={"capabilities": ["cockpit", "training", "macros", "research", "colab"]}
        )

    async def _action_create_wizard(self) -> MacroResult:
        """Start macro creation wizard."""
        return MacroResult(
            success=True,
            message="Starting macro creation wizard",
            spoken_response="Let's create a new macro. First, what trigger phrase would you like to use?",
            data={"wizard": "create", "step": 1}
        )

    async def _action_delete_wizard(self) -> MacroResult:
        """Start macro deletion wizard."""
        return MacroResult(
            success=True,
            message="Starting macro deletion wizard",
            spoken_response="Which macro would you like to delete? Say the trigger phrase.",
            data={"wizard": "delete", "step": 1}
        )

    async def _action_colab_offload(self) -> MacroResult:
        """Offload notebook to Colab."""
        # This would integrate with the ColabOffload widget
        return MacroResult(
            success=True,
            message="Colab offload initiated",
            spoken_response="I'm preparing to offload your work to Colab. Give me a moment.",
            data={"action": "colab_offload", "status": "initiated"}
        )

    async def _action_sync_notebooks(self) -> MacroResult:
        """Sync notebooks with Drive."""
        return MacroResult(
            success=True,
            message="Notebook sync initiated",
            spoken_response="Syncing your notebooks with Google Drive.",
            data={"action": "sync_notebooks", "status": "initiated"}
        )

    # =========================================================================
    # Macro Management
    # =========================================================================

    def add_macro(
        self,
        trigger: str,
        macro_type: str,
        command: Any,
        description: str,
        speak_summary: str
    ) -> bool:
        """
        Add a new macro.

        Args:
            trigger: Trigger phrase
            macro_type: Type of macro
            command: Command to execute
            description: Technical description
            speak_summary: How to explain to user

        Returns:
            True if added successfully
        """
        trigger = trigger.lower().strip()

        if trigger in self.macros:
            logger.warning(f"Macro already exists: {trigger}")
            return False

        self.macros[trigger] = {
            "type": macro_type,
            "command": command,
            "description": description,
            "speak_summary": speak_summary
        }

        # Save to file
        self._save_macros()
        logger.info(f"Added macro: {trigger}")
        return True

    def delete_macro(self, trigger: str) -> bool:
        """
        Delete a macro.

        Args:
            trigger: Trigger phrase

        Returns:
            True if deleted
        """
        trigger = trigger.lower().strip()

        if trigger not in self.macros:
            logger.warning(f"Macro not found: {trigger}")
            return False

        del self.macros[trigger]

        # Save to file
        self._save_macros()
        logger.info(f"Deleted macro: {trigger}")
        return True

    def _save_macros(self):
        """Save macros to configuration file."""
        try:
            # Load existing config to preserve version and comments
            if self.config_path.exists():
                with open(self.config_path) as f:
                    config = yaml.safe_load(f) or {}
            else:
                config = {"version": 1}

            config["macros"] = self.macros

            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            logger.info(f"Saved {len(self.macros)} macros to {self.config_path}")

        except Exception as e:
            logger.error(f"Error saving macros: {e}")

    def explain_macro(self, trigger: str) -> Optional[str]:
        """
        Get explanation for a macro.

        Args:
            trigger: Trigger phrase

        Returns:
            Speak summary or None
        """
        trigger = trigger.lower().strip()
        macro = self.macros.get(trigger)

        if macro:
            return macro.get("speak_summary", macro.get("description", "No explanation available."))

        return None

    def get_macro(self, trigger: str) -> Optional[Dict[str, Any]]:
        """
        Get macro by trigger phrase.

        Args:
            trigger: Trigger phrase

        Returns:
            Macro dict or None
        """
        return self.macros.get(trigger.lower().strip())

    @property
    def current_mode(self) -> str:
        """Get current Ara mode."""
        return self._current_mode

    @property
    def current_avatar(self) -> Dict[str, str]:
        """Get current avatar configuration."""
        return self._current_avatar.copy()

    async def cleanup(self):
        """Clean up resources."""
        await self.tfan_client.cleanup()
        logger.info("VoiceMacroProcessor cleaned up")


# Convenience function for processing voice input
async def process_voice_command(
    voice_input: str,
    config_path: str = "config/voice_macros.yaml",
    tfan_url: str = "http://localhost:8080"
) -> MacroResult:
    """
    Process a voice command through the macro system.

    Args:
        voice_input: Voice input text
        config_path: Path to macros config
        tfan_url: T-FAN API URL

    Returns:
        MacroResult
    """
    processor = VoiceMacroProcessor(
        config_path=config_path,
        tfan_base_url=tfan_url
    )

    match = processor.match_macro(voice_input)
    result = await processor.execute_macro(match)

    await processor.cleanup()
    return result
