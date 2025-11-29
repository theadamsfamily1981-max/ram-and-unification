#!/usr/bin/env python3
"""
Standalone Setup Server for Ara Avatar System
Simple web interface without complex dependencies
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
import subprocess
import shutil
import os

app = FastAPI(title="Ara Avatar Setup")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Main setup page"""
    return open("/home/user/ram-and-unification/setup_page.html").read()

@app.get("/api/system/status")
async def system_status():
    """Get comprehensive system status"""
    home_dir = Path.home()
    base_dir = home_dir / "tfan-ara-system"

    status = {
        "avatar_system": {
            "installed": Path("/home/user/ram-and-unification").exists(),
            "path": "/home/user/ram-and-unification"
        },
        "tfan_cockpit": {
            "installed": (base_dir / "Quanta-meis-nib-cis").exists(),
            "path": str(base_dir / "Quanta-meis-nib-cis")
        },
        "ollama": {
            "installed": shutil.which("ollama") is not None,
            "running": False
        },
        "dependencies": {
            "python": shutil.which("python3") is not None,
            "ffmpeg": shutil.which("ffmpeg") is not None,
            "git": shutil.which("git") is not None
        },
        "base_directory": str(base_dir),
        "home_directory": str(home_dir)
    }

    # Check if Ollama is running
    if status["ollama"]["installed"]:
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                timeout=5
            )
            status["ollama"]["running"] = result.returncode == 0
        except:
            pass

    return JSONResponse(status)

@app.post("/api/system/install-dependencies")
async def install_dependencies():
    """Install system dependencies"""
    try:
        script_path = Path("/home/user/ram-and-unification/install_dependencies.sh")

        if not script_path.exists():
            raise HTTPException(404, detail="Installation script not found")

        process = subprocess.Popen(
            ["bash", str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        output_lines = []
        for line in iter(process.stdout.readline, ''):
            if line:
                output_lines.append(line.strip())

        process.wait()

        return JSONResponse({
            "success": process.returncode == 0,
            "output": "\n".join(output_lines[-50:]),
            "exit_code": process.returncode
        })
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/api/system/install-tfan")
async def install_tfan():
    """Download and install T-FAN"""
    try:
        home_dir = Path.home()
        base_dir = home_dir / "tfan-ara-system"
        tfan_dir = base_dir / "Quanta-meis-nib-cis"

        base_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = subprocess.run(
                ["git", "clone",
                 "https://github.com/theadamsfamily1981-max/Quanta-meis-nib-cis",
                 str(tfan_dir)],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                return JSONResponse({
                    "success": True,
                    "message": "T-FAN downloaded successfully",
                    "path": str(tfan_dir)
                })
            else:
                # Try alternate locations
                alt_paths = [
                    home_dir / "Quanta-meis-nib-cis-main",
                    home_dir / "Quanta-meis-nib-cis",
                    Path("/home/user/Quanta-meis-nib-cis-main"),
                    Path("/home/user/Quanta-meis-nib-cis")
                ]

                for alt_path in alt_paths:
                    if alt_path.exists():
                        shutil.copytree(alt_path, tfan_dir, dirs_exist_ok=True)
                        return JSONResponse({
                            "success": True,
                            "message": f"T-FAN copied from {alt_path}",
                            "path": str(tfan_dir)
                        })

                return JSONResponse({
                    "success": False,
                    "message": "Could not download T-FAN. Please download manually.",
                    "manual_url": "https://github.com/theadamsfamily1981-max/Quanta-meis-nib-cis",
                    "install_path": str(tfan_dir)
                })

        except subprocess.TimeoutExpired:
            return JSONResponse({
                "success": False,
                "message": "Download timed out. Check your internet connection."
            })

    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/api/system/fix-webkit")
async def fix_webkit():
    """Fix WebKit compatibility"""
    try:
        script_path = Path("/home/user/ram-and-unification/fix_webkit_now.sh")

        if not script_path.exists():
            raise HTTPException(404, detail="WebKit fix script not found")

        result = subprocess.run(
            ["bash", str(script_path)],
            capture_output=True,
            text=True,
            timeout=30
        )

        return JSONResponse({
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else None
        })

    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/api/system/launch/{component}")
async def launch_component(component: str):
    """Launch a component"""
    try:
        commands = {
            "api": ["python3", "-m", "src.main"],
            "voice": ["python3", "ara_voice_interface.py"],
            "tfan": ["tfan-gnome"],
        }

        if component not in commands:
            raise HTTPException(400, detail=f"Unknown component: {component}")

        cmd = commands[component]
        cwd = Path("/home/user/ram-and-unification")

        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True
        )

        return JSONResponse({
            "success": True,
            "message": f"{component} launched",
            "pid": process.pid
        })

    except Exception as e:
        raise HTTPException(500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("=" * 72)
    print("  🚀 ARA AVATAR SYSTEM SETUP")
    print("=" * 72)
    print()
    print("  Setup wizard running at:")
    print("  👉 http://localhost:8000")
    print()
    print("  Press CTRL+C to stop")
    print("=" * 72)
    print()
    uvicorn.run(app, host="0.0.0.0", port=8000)
