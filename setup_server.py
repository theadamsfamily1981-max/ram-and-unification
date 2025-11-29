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
    return get_setup_html()

def get_setup_html() -> str:
    """Get the setup wizard HTML."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ara Avatar System Setup</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem;
        }
        .wizard {
            background: white;
            border-radius: 1rem;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 800px;
            width: 100%;
            overflow: hidden;
        }
        .wizard-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            text-align: center;
        }
        .wizard-header h1 {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        .wizard-header p {
            opacity: 0.9;
            font-size: 1.1rem;
        }
        .wizard-body {
            padding: 2rem;
        }
        .step {
            display: none;
        }
        .step.active {
            display: block;
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin: 2rem 0;
        }
        .status-card {
            background: #f8f9fa;
            border: 2px solid #e9ecef;
            border-radius: 0.5rem;
            padding: 1.5rem;
        }
        .status-card.installed {
            border-color: #28a745;
            background: #d4edda;
        }
        .status-card.missing {
            border-color: #ffc107;
            background: #fff3cd;
        }
        .status-card h3 {
            font-size: 1rem;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .status-card p {
            font-size: 0.875rem;
            color: #666;
        }
        .status-icon {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            display: inline-block;
        }
        .status-icon.ok { background: #28a745; }
        .status-icon.warn { background: #ffc107; }
        .status-icon.error { background: #dc3545; }
        .btn {
            display: inline-block;
            padding: 0.75rem 2rem;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 0.5rem;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s;
            text-decoration: none;
        }
        .btn:hover {
            background: #5568d3;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        .btn-group {
            display: flex;
            gap: 1rem;
            margin-top: 2rem;
            justify-content: center;
        }
        .btn-secondary {
            background: #6c757d;
        }
        .btn-secondary:hover {
            background: #5a6268;
        }
        .btn-success {
            background: #28a745;
        }
        .btn-success:hover {
            background: #218838;
        }
        .progress-bar {
            background: #e9ecef;
            border-radius: 1rem;
            height: 8px;
            margin: 2rem 0;
            overflow: hidden;
        }
        .progress-fill {
            background: linear-gradient(90deg, #667eea, #764ba2);
            height: 100%;
            transition: width 0.3s;
        }
        .log-output {
            background: #1a1a1a;
            color: #0f0;
            font-family: 'Courier New', monospace;
            padding: 1rem;
            border-radius: 0.5rem;
            max-height: 300px;
            overflow-y: auto;
            font-size: 0.875rem;
            margin: 1rem 0;
        }
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .install-item {
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0.25rem;
        }
        .install-item h4 {
            margin-bottom: 0.5rem;
        }
        .install-item button {
            margin-top: 0.5rem;
        }
    </style>
</head>
<body>
    <div class="wizard">
        <div class="wizard-header">
            <h1>🚀 Ara Avatar System Setup</h1>
            <p>Complete installation and configuration wizard</p>
        </div>

        <div class="wizard-body">
            <!-- Step 1: System Check -->
            <div class="step active" id="step1">
                <h2>Step 1: System Status</h2>
                <p>Checking your system...</p>

                <div class="progress-bar">
                    <div class="progress-fill" style="width: 25%"></div>
                </div>

                <div class="status-grid" id="statusGrid">
                    <div class="status-card">
                        <h3><span class="loading"></span> Checking...</h3>
                    </div>
                </div>

                <div class="btn-group">
                    <button class="btn" onclick="checkSystem()">Refresh Status</button>
                    <button class="btn btn-success" onclick="goToStep(2)" id="continueBtn" disabled>Continue →</button>
                </div>
            </div>

            <!-- Step 2: Install Missing Components -->
            <div class="step" id="step2">
                <h2>Step 2: Install Components</h2>
                <p>Install missing components for full functionality</p>

                <div class="progress-bar">
                    <div class="progress-fill" style="width: 50%"></div>
                </div>

                <div id="installItems"></div>

                <div class="btn-group">
                    <button class="btn btn-secondary" onclick="goToStep(1)">← Back</button>
                    <button class="btn btn-success" onclick="goToStep(3)">Continue →</button>
                </div>
            </div>

            <!-- Step 3: Launch -->
            <div class="step" id="step3">
                <h2>Step 3: Ready to Launch!</h2>
                <p>Your Ara Avatar System is ready to use</p>

                <div class="progress-bar">
                    <div class="progress-fill" style="width: 100%"></div>
                </div>

                <div class="status-grid">
                    <div class="status-card installed">
                        <h3>🎤 Voice Interface</h3>
                        <p>Talk to Ara with voice commands</p>
                        <button class="btn" onclick="launch('voice')" style="margin-top: 1rem; font-size: 0.875rem;">Launch</button>
                    </div>
                    <div class="status-card installed">
                        <h3>🌐 Web Chat</h3>
                        <p>Multi-AI chat interface</p>
                        <button class="btn" onclick="alert('Web chat coming soon!')" style="margin-top: 1rem; font-size: 0.875rem;">Coming Soon</button>
                    </div>
                    <div class="status-card installed">
                        <h3>🖥️ T-FAN Cockpit</h3>
                        <p>Spaceship-style HUD interface</p>
                        <button class="btn" onclick="launch('tfan')" style="margin-top: 1rem; font-size: 0.875rem;">Launch</button>
                    </div>
                    <div class="status-card installed">
                        <h3>🎬 Avatar Generator</h3>
                        <p>Create talking avatar videos</p>
                        <button class="btn" onclick="launch('api')" style="margin-top: 1rem; font-size: 0.875rem;">Start API</button>
                    </div>
                </div>

                <div class="btn-group">
                    <button class="btn btn-secondary" onclick="goToStep(1)">Run Setup Again</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        let systemStatus = {};

        async function checkSystem() {
            try {
                const response = await fetch('/api/system/status');
                systemStatus = await response.json();
                displayStatus();
            } catch (error) {
                alert('Error checking system: ' + error.message);
            }
        }

        function displayStatus() {
            const grid = document.getElementById('statusGrid');
            const continueBtn = document.getElementById('continueBtn');

            const items = [
                {
                    name: 'Avatar System',
                    status: systemStatus.avatar_system?.installed,
                    icon: '🎭'
                },
                {
                    name: 'T-FAN Cockpit',
                    status: systemStatus.tfan_cockpit?.installed,
                    icon: '🖥️'
                },
                {
                    name: 'Ollama (AI)',
                    status: systemStatus.ollama?.installed,
                    icon: '🤖'
                },
                {
                    name: 'Dependencies',
                    status: systemStatus.dependencies?.python && systemStatus.dependencies?.ffmpeg,
                    icon: '📦'
                }
            ];

            grid.innerHTML = items.map(item => `
                <div class="status-card ${item.status ? 'installed' : 'missing'}">
                    <h3>
                        <span class="status-icon ${item.status ? 'ok' : 'warn'}"></span>
                        ${item.icon} ${item.name}
                    </h3>
                    <p>${item.status ? 'Installed' : 'Not installed'}</p>
                </div>
            `).join('');

            continueBtn.disabled = false;
        }

        function goToStep(stepNum) {
            document.querySelectorAll('.step').forEach(s => s.classList.remove('active'));
            document.getElementById('step' + stepNum).classList.add('active');

            if (stepNum === 2) {
                showInstallOptions();
            }
        }

        function showInstallOptions() {
            const container = document.getElementById('installItems');
            const options = [];

            if (!systemStatus.dependencies?.ffmpeg || !systemStatus.ollama?.installed) {
                options.push({
                    title: 'System Dependencies',
                    desc: 'Install FFmpeg, Ollama, and other required packages',
                    action: 'installDeps'
                });
            }

            if (!systemStatus.tfan_cockpit?.installed) {
                options.push({
                    title: 'T-FAN Cockpit',
                    desc: 'Download the T-FAN spaceship interface from GitHub',
                    action: 'installTFAN'
                });
            }

            if (systemStatus.tfan_cockpit?.installed) {
                options.push({
                    title: 'Fix WebKit',
                    desc: 'Fix WebKit compatibility issues in T-FAN',
                    action: 'fixWebKit'
                });
            }

            if (options.length === 0) {
                container.innerHTML = '<p style="text-align: center; color: #28a745; font-size: 1.2rem;">✅ All components installed!</p>';
            } else {
                container.innerHTML = options.map(opt => `
                    <div class="install-item">
                        <h4>${opt.title}</h4>
                        <p>${opt.desc}</p>
                        <button class="btn" onclick="${opt.action}()" id="btn_${opt.action}">Install</button>
                        <div id="log_${opt.action}" class="log-output" style="display: none;"></div>
                    </div>
                `).join('');
            }
        }

        async function installDeps() {
            const btn = document.getElementById('btn_installDeps');
            const log = document.getElementById('log_installDeps');

            btn.disabled = true;
            btn.innerHTML = '<span class="loading"></span> Installing...';
            log.style.display = 'block';
            log.textContent = 'Starting dependency installation...\\n';

            try {
                const response = await fetch('/api/system/install-dependencies', { method: 'POST' });
                const result = await response.json();

                log.textContent += result.output;

                if (result.success) {
                    btn.innerHTML = '✅ Installed';
                    btn.className = 'btn btn-success';
                } else {
                    btn.innerHTML = '❌ Failed';
                    btn.disabled = false;
                }
            } catch (error) {
                log.textContent += '\\nError: ' + error.message;
                btn.innerHTML = '❌ Error';
                btn.disabled = false;
            }
        }

        async function installTFAN() {
            const btn = document.getElementById('btn_installTFAN');
            const log = document.getElementById('log_installTFAN');

            btn.disabled = true;
            btn.innerHTML = '<span class="loading"></span> Downloading...';
            log.style.display = 'block';
            log.textContent = 'Downloading T-FAN from GitHub...\\n';

            try {
                const response = await fetch('/api/system/install-tfan', { method: 'POST' });
                const result = await response.json();

                log.textContent += result.message + '\\n';

                if (result.success) {
                    btn.innerHTML = '✅ Downloaded';
                    btn.className = 'btn btn-success';
                } else {
                    btn.innerHTML = '⚠️ Manual Download Required';
                    if (result.manual_url) {
                        log.textContent += '\\nPlease download from: ' + result.manual_url;
                        log.textContent += '\\nExtract to: ' + result.install_path;
                    }
                }
            } catch (error) {
                log.textContent += '\\nError: ' + error.message;
                btn.innerHTML = '❌ Error';
                btn.disabled = false;
            }
        }

        async function fixWebKit() {
            const btn = document.getElementById('btn_fixWebKit');
            const log = document.getElementById('log_fixWebKit');

            btn.disabled = true;
            btn.innerHTML = '<span class="loading"></span> Fixing...';
            log.style.display = 'block';
            log.textContent = 'Fixing WebKit compatibility...\\n';

            try {
                const response = await fetch('/api/system/fix-webkit', { method: 'POST' });
                const result = await response.json();

                log.textContent += result.output;

                if (result.success) {
                    btn.innerHTML = '✅ Fixed';
                    btn.className = 'btn btn-success';
                } else {
                    btn.innerHTML = '❌ Failed';
                    log.textContent += '\\nError: ' + result.error;
                }
            } catch (error) {
                log.textContent += '\\nError: ' + error.message;
                btn.innerHTML = '❌ Error';
                btn.disabled = false;
            }
        }

        async function launch(component) {
            try {
                const response = await fetch(`/api/system/launch/${component}`, { method: 'POST' });
                const result = await response.json();

                if (result.success) {
                    alert(`${component} launched successfully! PID: ${result.pid}`);
                } else {
                    alert(`Failed to launch ${component}`);
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }

        // Auto-check on load
        checkSystem();
    </script>
</body>
</html>
    """

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
