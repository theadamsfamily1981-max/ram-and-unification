#!/usr/bin/env python3
"""Quick test script to verify all components are working."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from src.audio_input import AudioRecorder, VoiceActivityDetector
        from src.asr import WhisperASR
        from src.dialogue import DialogueManager
        from src.tts import CoquiTTS
        print("✅ All imports successful!")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_audio():
    """Test audio input."""
    print("\nTesting audio devices...")
    try:
        from src.audio_input import AudioRecorder
        recorder = AudioRecorder()
        print("✅ Audio recorder initialized")
        print("\nAvailable audio devices:")
        recorder.list_devices()
        return True
    except Exception as e:
        print(f"❌ Audio test failed: {e}")
        return False


def test_asr():
    """Test ASR."""
    print("\nTesting ASR...")
    try:
        from src.asr import WhisperASR
        print("Loading Whisper model (this may take a minute)...")
        asr = WhisperASR(model_name="tiny", device="cpu")
        print("✅ ASR initialized successfully")
        return True
    except Exception as e:
        print(f"❌ ASR test failed: {e}")
        print("Note: Whisper model will download on first use")
        return False


def test_dialogue():
    """Test dialogue manager."""
    print("\nTesting Dialogue Manager...")
    try:
        from src.dialogue import DialogueManager
        print("Connecting to Ollama...")
        dm = DialogueManager(engine="ollama", model="llama3.2")
        print("Sending test message...")
        response = dm.get_response("Say hello!")
        print(f"Response: {response}")
        print("✅ Dialogue manager working!")
        return True
    except Exception as e:
        print(f"❌ Dialogue test failed: {e}")
        print("Note: Make sure Ollama is running: ollama serve")
        print("And model is downloaded: ollama pull llama3.2")
        return False


def test_tts():
    """Test TTS."""
    print("\nTesting TTS...")
    try:
        from src.tts import CoquiTTS
        print("Loading TTS model (this may take a minute)...")
        tts = CoquiTTS(device="cpu")
        print("✅ TTS initialized successfully")
        return True
    except Exception as e:
        print(f"❌ TTS test failed: {e}")
        print("Note: TTS model will download on first use")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Conversational Avatar AI - Component Test")
    print("=" * 60 + "\n")

    results = []

    results.append(("Imports", test_imports()))
    results.append(("Audio", test_audio()))
    results.append(("ASR", test_asr()))
    results.append(("Dialogue", test_dialogue()))
    results.append(("TTS", test_tts()))

    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:20} {status}")

    all_passed = all(passed for _, passed in results)

    print("=" * 60)
    if all_passed:
        print("🎉 All tests passed! You're ready to run: python main.py")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        print("See INSTALL.md for troubleshooting steps.")
    print("=" * 60 + "\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
