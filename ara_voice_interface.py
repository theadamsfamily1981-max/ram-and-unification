#!/usr/bin/env python3
"""Ara Voice Interface - Voice-controlled AI co-pilot with talking avatar.

This is the main interface for interacting with Ara through voice commands.
It integrates:
- Speech recognition (listening for voice input)
- Voice macro processing (executing commands)
- Ara avatar backend (generating responses with talking avatar)
- T-FAN cockpit control
- Text-to-speech with Ara persona

Usage:
    python3 ara_voice_interface.py                 # Start voice mode
    python3 ara_voice_interface.py --text-only     # Text chat mode (no voice)
    python3 ara_voice_interface.py --test "hello"  # Test with text input
"""

import asyncio
import sys
import argparse
from pathlib import Path
from typing import Optional

# Add multi-ai-workspace to path
sys.path.insert(0, str(Path(__file__).parent / "multi-ai-workspace"))

from src.integrations.ara_avatar_backend import AraAvatarBackend
from src.widgets.voice_macros import VoiceMacroProcessor, MacroType
from src.integrations.tfan_client import TFANClient
from src.core.backend import Context


class AraVoiceInterface:
    """
    Ara Voice Interface.

    Provides voice-controlled interaction with Ara, including:
    - Voice command recognition
    - Voice macro execution
    - Talking avatar video generation
    - Text-to-speech responses
    - T-FAN cockpit integration
    """

    def __init__(
        self,
        voice_enabled: bool = True,
        avatar_enabled: bool = True,
        tfan_url: str = "http://localhost:8080"
    ):
        """
        Initialize Ara Voice Interface.

        Args:
            voice_enabled: Enable voice input/output
            avatar_enabled: Enable avatar video generation
            tfan_url: T-FAN cockpit API URL
        """
        self.voice_enabled = voice_enabled
        self.avatar_enabled = avatar_enabled

        # Initialize Ara backend (uses OLLAMA_MODEL from .env, defaults to 'ara')
        print("🤖 Initializing Ara avatar backend...")
        self.ara = AraAvatarBackend(name="Ara")

        # Initialize voice macro processor
        print("🎙️  Loading voice macros...")
        self.macro_processor = VoiceMacroProcessor(
            config_path="multi-ai-workspace/config/voice_macros.yaml",
            tfan_base_url=tfan_url
        )

        # Conversation context
        self.context = Context()
        self.context.system_prompt = self.ara._get_system_prompt()

        # Voice recognition (lazy loaded)
        self.recognizer = None

        print(f"✨ Ara is online and ready! (using model: {self.ara.ollama_model})")
        self._speak_greeting()

    def _speak_greeting(self):
        """Speak Ara's greeting message."""
        greeting = "Hey, you. I'm online, systems are stable, and you look like you need a win. Where do you want to start?"
        print(f"\n💬 Ara: {greeting}\n")

        if self.voice_enabled:
            self._speak(greeting)

    def _speak(self, text: str):
        """Convert text to speech and play it."""
        try:
            import subprocess

            # Use espeak for quick TTS (can be replaced with better TTS)
            subprocess.run(
                ["espeak-ng", "-s", "140", "-p", "30", text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(f"⚠️  TTS error: {e}")

    async def listen_for_voice_input(self) -> Optional[str]:
        """
        Listen for voice input using speech recognition.

        Returns:
            Recognized text or None
        """
        if not self.voice_enabled:
            return None

        try:
            # Lazy import speech recognition
            if self.recognizer is None:
                import speech_recognition as sr
                self.recognizer = sr.Recognizer()

            with sr.Microphone() as source:
                print("🎧 Listening... (speak now)")
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=15)

            print("🔄 Processing...")
            text = self.recognizer.recognize_google(audio)
            print(f"✅ Recognized: {text}")
            return text

        except Exception as e:
            print(f"⚠️  Voice recognition error: {e}")
            return None

    async def process_input(self, user_input: str) -> dict:
        """
        Process user input through voice macros or regular chat.

        Args:
            user_input: User input text

        Returns:
            Response dictionary
        """
        # Check if input matches a voice macro
        macro_match = self.macro_processor.match_macro(user_input)

        if macro_match.matched:
            print(f"🎯 Macro matched: {macro_match.macro_name} (confidence: {macro_match.confidence:.2f})")

            # Execute macro
            result = await self.macro_processor.execute_macro(macro_match)

            response = {
                "type": "macro",
                "macro": macro_match.macro_name,
                "text": result.spoken_response,
                "success": result.success,
                "data": result.data
            }

            # Update Ara state if mode/avatar changed
            if macro_match.macro_type == MacroType.ARA_MODE:
                mode = macro_match.command
                self.ara.set_mode(mode)

            elif macro_match.macro_type == MacroType.ARA_AVATAR:
                config = macro_match.command
                if isinstance(config, dict):
                    profile = config.get("profile", "default")
                    mood = config.get("mood", "neutral")
                    self.ara.set_avatar_profile(profile, mood)

            return response

        else:
            # Regular chat - send to Ara
            print("💭 Processing with Ara...")

            if self.avatar_enabled:
                # Generate full avatar response with video
                result = await self.ara.generate_avatar_response(
                    prompt=user_input,
                    context=self.context,
                    use_tts=self.voice_enabled
                )

                # Update conversation history
                self.context.conversation_history.append({"role": "user", "content": user_input})
                self.context.conversation_history.append({"role": "assistant", "content": result["text"]})

                return {
                    "type": "chat",
                    "text": result["text"],
                    "audio_path": result.get("audio_path"),
                    "video_path": result.get("video_path"),
                    "error": result.get("error")
                }

            else:
                # Text-only response
                response = await self.ara.send_message(
                    prompt=user_input,
                    context=self.context
                )

                # Update conversation history
                self.context.conversation_history.append({"role": "user", "content": user_input})
                self.context.conversation_history.append({"role": "assistant", "content": response.content})

                return {
                    "type": "chat",
                    "text": response.content,
                    "error": response.error
                }

    async def run_voice_loop(self):
        """Run the main voice interaction loop."""
        print("\n" + "="*60)
        print("🎙️  ARA VOICE MODE ACTIVE")
        print("="*60)
        print("Say 'Ara' to wake up, then give your command.")
        print("Say 'exit' or 'quit' to stop.")
        print("=" * 60 + "\n")

        while True:
            try:
                # Listen for voice input
                user_input = await self.listen_for_voice_input()

                if not user_input:
                    continue

                # Check for exit commands
                if user_input.lower() in ["exit", "quit", "stop", "goodbye"]:
                    farewell = "Shutting down. It's been good working with you."
                    print(f"\n💬 Ara: {farewell}\n")
                    self._speak(farewell)
                    break

                # Process input
                response = await self.process_input(user_input)

                # Display and speak response
                print(f"\n💬 Ara: {response['text']}\n")

                if self.voice_enabled and response.get("audio_path"):
                    # Play generated TTS audio
                    try:
                        import subprocess
                        subprocess.run(["aplay", response["audio_path"]], check=False)
                    except:
                        # Fallback to espeak
                        self._speak(response["text"])
                elif self.voice_enabled:
                    self._speak(response["text"])

                # Play avatar video if generated
                if response.get("video_path"):
                    print(f"🎬 Avatar video: {response['video_path']}")
                    # Optionally auto-play video
                    # subprocess.Popen(["vlc", response["video_path"]])

            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user")
                break
            except Exception as e:
                print(f"\n⚠️  Error: {e}\n")

    async def run_text_loop(self):
        """Run text-based chat loop."""
        print("\n" + "="*60)
        print("💬 ARA TEXT CHAT MODE")
        print("="*60)
        print("Type your messages and press Enter.")
        print("Type 'exit' or 'quit' to stop.")
        print("="*60 + "\n")

        while True:
            try:
                # Get text input
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                # Check for exit
                if user_input.lower() in ["exit", "quit", "stop"]:
                    print("\n💬 Ara: Shutting down. It's been good working with you.\n")
                    break

                # Process input
                response = await self.process_input(user_input)

                # Display response
                print(f"\n💬 Ara: {response['text']}\n")

                # Show video path if generated
                if response.get("video_path"):
                    print(f"🎬 Avatar video: {response['video_path']}\n")

            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user")
                break
            except Exception as e:
                print(f"\n⚠️  Error: {e}\n")

    async def cleanup(self):
        """Clean up resources."""
        await self.macro_processor.cleanup()
        print("✅ Ara shut down cleanly")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Ara Voice Interface - Your AI Co-Pilot")
    parser.add_argument("--text-only", action="store_true", help="Disable voice input/output")
    parser.add_argument("--no-avatar", action="store_true", help="Disable avatar video generation")
    parser.add_argument("--test", type=str, help="Test with a text input")
    parser.add_argument("--tfan-url", type=str, default="http://localhost:8080", help="T-FAN API URL")

    args = parser.parse_args()

    # Create interface (model auto-detected from OLLAMA_MODEL in .env, defaults to 'ara')
    interface = AraVoiceInterface(
        voice_enabled=not args.text_only,
        avatar_enabled=not args.no_avatar,
        tfan_url=args.tfan_url
    )

    try:
        if args.test:
            # Test mode - process single input
            print(f"\n🧪 Test mode: {args.test}\n")
            response = await interface.process_input(args.test)
            print(f"\n💬 Ara: {response['text']}\n")

            if response.get("video_path"):
                print(f"🎬 Avatar video: {response['video_path']}\n")

        elif args.text_only:
            # Text chat mode
            await interface.run_text_loop()
        else:
            # Voice mode
            await interface.run_voice_loop()

    finally:
        await interface.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
