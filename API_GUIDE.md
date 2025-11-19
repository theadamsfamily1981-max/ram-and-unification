# Talking Avatar API - Complete API Guide

## Table of Contents
- [Overview](#overview)
- [Authentication](#authentication)
- [Endpoints](#endpoints)
- [Request/Response Examples](#requestresponse-examples)
- [Error Handling](#error-handling)
- [Rate Limits](#rate-limits)

## Overview

Base URL: `http://localhost:8000/api/v1`

All endpoints return JSON responses unless otherwise specified.

## Authentication

Currently, authentication is optional. To enable API key authentication:

1. Set `API_KEY` in your `.env` file
2. Include the API key in requests:
   - Header: `X-API-Key: your_key_here`
   - Or query parameter: `?api_key=your_key_here`

## Endpoints

### 1. Health Check

Check if the API is running and get system information.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "device": "cpu"
}
```

---

### 2. Upload Image

Upload an image file to use for avatar generation.

**Endpoint:** `POST /upload/image`

**Content-Type:** `multipart/form-data`

**Parameters:**
- `file` (required): Image file (PNG, JPG, JPEG)

**Response:**
```json
{
  "filename": "abc123.jpg",
  "message": "Image uploaded successfully"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/upload/image" \
  -F "file=@/path/to/image.jpg"
```

---

### 3. Upload Audio

Upload an audio file to use for avatar generation.

**Endpoint:** `POST /upload/audio`

**Content-Type:** `multipart/form-data`

**Parameters:**
- `file` (required): Audio file (WAV, MP3)

**Response:**
```json
{
  "filename": "def456.wav",
  "message": "Audio uploaded successfully"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/upload/audio" \
  -F "file=@/path/to/audio.wav"
```

---

### 4. Generate Avatar (Synchronous)

Generate a talking avatar video synchronously. This endpoint waits for the generation to complete before returning.

**Endpoint:** `POST /generate`

**Content-Type:** `application/json`

**Request Body:**
```json
{
  "image_filename": "abc123.jpg",
  "audio_filename": "def456.wav",
  "output_fps": 25,
  "output_resolution": 512
}
```

**Parameters:**
- `image_filename` (required): Filename of uploaded image
- `audio_filename` (required): Filename of uploaded audio
- `output_fps` (optional): Output video FPS (10-60, default: 25)
- `output_resolution` (optional): Output resolution (256, 512, or 1024, default: 512)

**Response:**
```json
{
  "success": true,
  "video_url": "/download/xyz789.mp4",
  "video_filename": "xyz789.mp4",
  "duration": 15.3,
  "frames_generated": 383,
  "error_message": null
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "image_filename": "abc123.jpg",
    "audio_filename": "def456.wav",
    "output_fps": 25,
    "output_resolution": 512
  }'
```

---

### 5. Generate Avatar (Asynchronous)

Start an async avatar generation job. Returns immediately with a job ID for tracking.

**Endpoint:** `POST /generate/async`

**Content-Type:** `application/json`

**Request Body:** Same as synchronous generate

**Response:**
```json
{
  "job_id": "job_abc123xyz",
  "status": "pending",
  "progress": 0,
  "result": null
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/generate/async" \
  -H "Content-Type: application/json" \
  -d '{
    "image_filename": "abc123.jpg",
    "audio_filename": "def456.wav"
  }'
```

---

### 6. Check Job Status

Check the status of an async generation job.

**Endpoint:** `GET /status/{job_id}`

**Parameters:**
- `job_id` (path): Job ID from async generate

**Response (In Progress):**
```json
{
  "job_id": "job_abc123xyz",
  "status": "processing",
  "progress": 45.5,
  "result": null
}
```

**Response (Completed):**
```json
{
  "job_id": "job_abc123xyz",
  "status": "completed",
  "progress": 100,
  "result": {
    "success": true,
    "video_url": "/download/xyz789.mp4",
    "video_filename": "xyz789.mp4",
    "duration": 15.3,
    "frames_generated": 383,
    "error_message": null
  }
}
```

**Status Values:**
- `pending`: Job queued
- `processing`: Currently generating
- `completed`: Generation finished successfully
- `failed`: Generation failed

**cURL Example:**
```bash
curl "http://localhost:8000/api/v1/status/job_abc123xyz"
```

---

### 7. Download Video

Download a generated video file.

**Endpoint:** `GET /download/{filename}`

**Parameters:**
- `filename` (path): Video filename from generate response

**Response:** Video file (video/mp4)

**cURL Example:**
```bash
curl "http://localhost:8000/api/v1/download/xyz789.mp4" \
  -o output.mp4
```

---

### 8. Cleanup File

Delete an uploaded or generated file.

**Endpoint:** `DELETE /cleanup/{filename}`

**Parameters:**
- `filename` (path): File to delete

**Response:**
```json
{
  "message": "File xyz789.mp4 deleted successfully"
}
```

**cURL Example:**
```bash
curl -X DELETE "http://localhost:8000/api/v1/cleanup/xyz789.mp4"
```

---

## Request/Response Examples

### Complete Workflow Example (Python)

```python
import requests
import time

API_URL = "http://localhost:8000/api/v1"

# Step 1: Upload image
with open("avatar.jpg", "rb") as f:
    resp = requests.post(f"{API_URL}/upload/image", files={"file": f})
    image_filename = resp.json()["filename"]

# Step 2: Upload audio
with open("speech.wav", "rb") as f:
    resp = requests.post(f"{API_URL}/upload/audio", files={"file": f})
    audio_filename = resp.json()["filename"]

# Step 3: Start async generation
resp = requests.post(
    f"{API_URL}/generate/async",
    json={
        "image_filename": image_filename,
        "audio_filename": audio_filename,
        "output_fps": 30,
        "output_resolution": 512
    }
)
job_id = resp.json()["job_id"]

# Step 4: Poll for completion
while True:
    resp = requests.get(f"{API_URL}/status/{job_id}")
    status = resp.json()

    if status["status"] == "completed":
        video_url = status["result"]["video_url"]
        break
    elif status["status"] == "failed":
        print("Generation failed!")
        break

    print(f"Progress: {status['progress']}%")
    time.sleep(2)

# Step 5: Download video
resp = requests.get(f"{API_URL}{video_url}")
with open("output.mp4", "wb") as f:
    f.write(resp.content)

# Step 6: Cleanup (optional)
requests.delete(f"{API_URL}/cleanup/{image_filename}")
requests.delete(f"{API_URL}/cleanup/{audio_filename}")
```

---

## Error Handling

All errors return appropriate HTTP status codes and JSON error messages.

### Common Error Responses

**400 Bad Request:**
```json
{
  "detail": "Invalid file type. Allowed: image/png, image/jpeg, image/jpg"
}
```

**404 Not Found:**
```json
{
  "detail": "Image file not found: abc123.jpg"
}
```

**500 Internal Server Error:**
```json
{
  "detail": "Internal server error occurred"
}
```

### HTTP Status Codes

- `200 OK`: Request successful
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error

---

## Rate Limits

Currently no rate limits are enforced. For production use, consider implementing:
- Request rate limiting
- File size limits
- Concurrent job limits

---

## Best Practices

1. **File Management**: Delete uploaded files after use to save space
2. **Async for Long Jobs**: Use async endpoint for videos longer than 10 seconds
3. **Error Handling**: Always check the `success` field in responses
4. **Polling**: Poll status endpoint every 2-5 seconds, not more frequently
5. **File Formats**: Use WAV for best audio quality, MP3 for smaller files

---

## Support

For issues or questions:
- Check API documentation: http://localhost:8000/docs
- Review examples in `examples/` directory
- Open an issue on GitHub
