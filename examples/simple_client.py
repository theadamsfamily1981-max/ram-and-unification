"""Simple Python client example for Talking Avatar API."""

import requests
import sys
from pathlib import Path


def generate_avatar(image_path: str, audio_path: str, api_url: str = "http://localhost:8000/api/v1"):
    """Generate a talking avatar video.

    Args:
        image_path: Path to input image
        audio_path: Path to input audio
        api_url: Base API URL
    """
    print("=" * 60)
    print("Talking Avatar Generator")
    print("=" * 60)

    # Check files exist
    if not Path(image_path).exists():
        print(f"Error: Image file not found: {image_path}")
        return

    if not Path(audio_path).exists():
        print(f"Error: Audio file not found: {audio_path}")
        return

    try:
        # 1. Upload image
        print("\n1. Uploading image...")
        with open(image_path, "rb") as f:
            response = requests.post(
                f"{api_url}/upload/image",
                files={"file": f}
            )
            response.raise_for_status()
            image_filename = response.json()["filename"]
            print(f"   ✓ Image uploaded: {image_filename}")

        # 2. Upload audio
        print("\n2. Uploading audio...")
        with open(audio_path, "rb") as f:
            response = requests.post(
                f"{api_url}/upload/audio",
                files={"file": f}
            )
            response.raise_for_status()
            audio_filename = response.json()["filename"]
            print(f"   ✓ Audio uploaded: {audio_filename}")

        # 3. Generate avatar
        print("\n3. Generating talking avatar...")
        print("   (This may take a minute...)")
        response = requests.post(
            f"{api_url}/generate",
            json={
                "image_filename": image_filename,
                "audio_filename": audio_filename,
                "output_fps": 25,
                "output_resolution": 512
            }
        )
        response.raise_for_status()
        result = response.json()

        if result["success"]:
            print(f"   ✓ Generation successful!")
            print(f"   Duration: {result['duration']:.2f}s")
            print(f"   Frames: {result['frames_generated']}")

            # 4. Download video
            print("\n4. Downloading video...")
            video_url = result["video_url"]
            video_response = requests.get(f"{api_url}{video_url}")
            video_response.raise_for_status()

            output_file = "output_avatar.mp4"
            with open(output_file, "wb") as f:
                f.write(video_response.content)

            print(f"   ✓ Video saved to: {output_file}")
            print("\n" + "=" * 60)
            print("SUCCESS! Your talking avatar is ready!")
            print("=" * 60)

        else:
            print(f"   ✗ Generation failed: {result['error_message']}")

    except requests.exceptions.ConnectionError:
        print("\nError: Cannot connect to API server.")
        print("Make sure the server is running at:", api_url)
    except Exception as e:
        print(f"\nError: {str(e)}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python simple_client.py <image_path> <audio_path>")
        print("\nExample:")
        print("  python simple_client.py avatar.jpg speech.wav")
        sys.exit(1)

    image_path = sys.argv[1]
    audio_path = sys.argv[2]

    generate_avatar(image_path, audio_path)
