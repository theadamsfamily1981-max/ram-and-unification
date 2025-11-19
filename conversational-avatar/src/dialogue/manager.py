"""Dialogue manager with LLM integration."""

import ollama
from typing import List, Dict, Optional
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger(__name__)


class DialogueManager:
    """Manages conversation with LLM."""

    def __init__(
        self,
        engine: str = "ollama",
        model: str = "llama3.2",
        system_prompt: Optional[str] = None,
        max_history: int = 10,
        **kwargs
    ):
        """Initialize dialogue manager.

        Args:
            engine: LLM engine (ollama, openai, anthropic)
            model: Model name
            system_prompt: System prompt for the assistant
            max_history: Maximum conversation history to keep
            **kwargs: Additional engine-specific parameters
        """
        self.engine = engine
        self.model = model
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.max_history = max_history
        self.kwargs = kwargs

        self.conversation_history: List[Dict[str, str]] = []

        logger.info(f"DialogueManager initialized: {engine}/{model}")

        # Initialize engine-specific settings
        if engine == "ollama":
            self._init_ollama()
        elif engine == "openai":
            self._init_openai()
        else:
            logger.warning(f"Unsupported engine: {engine}, using ollama")
            self.engine = "ollama"
            self._init_ollama()

    def _default_system_prompt(self) -> str:
        """Get default system prompt."""
        return """You are a friendly and helpful AI assistant.
Keep your responses concise and conversational, as they will be spoken aloud.
Aim for responses under 3 sentences unless more detail is specifically requested.
Be warm, empathetic, and engaging in your tone."""

    def _init_ollama(self):
        """Initialize Ollama client."""
        try:
            # Check if model is available
            models = ollama.list()
            model_names = [m['name'] for m in models.get('models', [])]

            if not any(self.model in name for name in model_names):
                logger.warning(f"Model '{self.model}' not found. Available models: {model_names}")
                logger.info(f"Attempting to pull model '{self.model}'...")
                ollama.pull(self.model)

            logger.info(f"Ollama model '{self.model}' ready")

        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")
            logger.info("Make sure Ollama is installed and running: https://ollama.ai")
            raise

    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            import openai
            api_key = self.kwargs.get('api_key') or os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not provided")
            openai.api_key = api_key
            self.openai_client = openai
            logger.info("OpenAI client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
            raise

    def add_user_message(self, text: str):
        """Add user message to conversation history.

        Args:
            text: User's message
        """
        self.conversation_history.append({
            "role": "user",
            "content": text
        })
        self._trim_history()
        logger.info(f"User: {text}")

    def add_assistant_message(self, text: str):
        """Add assistant message to conversation history.

        Args:
            text: Assistant's message
        """
        self.conversation_history.append({
            "role": "assistant",
            "content": text
        })
        self._trim_history()
        logger.info(f"Assistant: {text}")

    def get_response(self, user_message: Optional[str] = None) -> str:
        """Get response from LLM.

        Args:
            user_message: Optional user message to add before getting response

        Returns:
            Assistant's response text
        """
        if user_message:
            self.add_user_message(user_message)

        try:
            if self.engine == "ollama":
                response = self._get_ollama_response()
            elif self.engine == "openai":
                response = self._get_openai_response()
            else:
                response = "I'm sorry, I don't understand."

            self.add_assistant_message(response)
            return response

        except Exception as e:
            logger.error(f"Failed to get response: {e}")
            error_msg = "I'm sorry, I encountered an error. Please try again."
            self.add_assistant_message(error_msg)
            return error_msg

    def _get_ollama_response(self) -> str:
        """Get response from Ollama."""
        # Prepare messages with system prompt
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)

        logger.info("Generating response with Ollama...")

        # Get response
        response = ollama.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": self.kwargs.get("temperature", 0.7),
                "top_p": self.kwargs.get("top_p", 0.9),
                "num_predict": self.kwargs.get("max_tokens", 150),
            }
        )

        return response['message']['content'].strip()

    def _get_openai_response(self) -> str:
        """Get response from OpenAI."""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)

        logger.info("Generating response with OpenAI...")

        response = self.openai_client.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=self.kwargs.get("temperature", 0.7),
            max_tokens=self.kwargs.get("max_tokens", 150),
            top_p=self.kwargs.get("top_p", 0.9)
        )

        return response.choices[0].message.content.strip()

    def _trim_history(self):
        """Trim conversation history to max length."""
        if len(self.conversation_history) > self.max_history:
            # Keep only recent messages
            self.conversation_history = self.conversation_history[-self.max_history:]
            logger.debug(f"Trimmed history to {self.max_history} messages")

    def clear_history(self, keep_last_n: int = 0):
        """Clear conversation history.

        Args:
            keep_last_n: Number of recent messages to keep
        """
        if keep_last_n > 0:
            self.conversation_history = self.conversation_history[-keep_last_n:]
        else:
            self.conversation_history = []
        logger.info(f"Cleared conversation history (kept last {keep_last_n})")

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history.

        Returns:
            List of conversation messages
        """
        return self.conversation_history.copy()

    def set_system_prompt(self, prompt: str):
        """Update system prompt.

        Args:
            prompt: New system prompt
        """
        self.system_prompt = prompt
        logger.info("System prompt updated")

    def export_conversation(self, output_path: Path):
        """Export conversation to file.

        Args:
            output_path: Output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write(f"System Prompt:\n{self.system_prompt}\n\n")
            f.write("=" * 60 + "\n\n")

            for msg in self.conversation_history:
                role = msg['role'].capitalize()
                content = msg['content']
                f.write(f"{role}: {content}\n\n")

        logger.info(f"Conversation exported to {output_path}")
