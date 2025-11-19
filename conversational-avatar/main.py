#!/usr/bin/env python3
"""
Conversational Talking Avatar AI - Main Entry Point
Phase 2: Minimal Voice Assistant Prototype (Audio Only)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.orchestrator import VoiceAssistantPrototype


def main():
    """Main entry point."""
    try:
        # Create and run voice assistant
        assistant = VoiceAssistantPrototype()
        assistant.run()

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Exiting...")
        sys.exit(0)

    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
