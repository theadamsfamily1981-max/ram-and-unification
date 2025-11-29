# 🚀 ARA AVATAR SYSTEM - START HERE

## MASTER CODE - OPTIMIZED & DEBUGGED

The setup system has been completely rewritten to work from **ANY directory** on **ANY user account**. All hardcoded paths have been eliminated.

---

## ⚡ QUICK START (Copy & Paste)

### Option 1: If you have the files already

```bash
cd ~/ram-and-unification-main  # Or wherever you cloned this
python3 setup_server.py
```

Then open your browser to: **http://localhost:8000**

---

### Option 2: Universal Quick Start (Works Anywhere)

```bash
bash <(curl -sL https://raw.githubusercontent.com/theadamsfamily1981-max/ram-and-unification/main/QUICK_START.sh)
```

This will:
- Auto-download the latest setup server
- Install dependencies (FastAPI, Uvicorn)
- Launch the web wizard
- Work from ANY directory!

---

### Option 3: Manual Setup

```bash
# 1. Install dependencies
sudo apt install -y python3-fastapi python3-uvicorn

# 2. Download setup server
wget https://raw.githubusercontent.com/theadamsfamily1981-max/ram-and-unification/main/setup_server.py

# 3. Run it
python3 setup_server.py
```

---

## 🌐 ACCESS THE WIZARD

Once the server starts, you'll see:

```
========================================================================
  🚀 ARA AVATAR SYSTEM SETUP (UNIVERSAL)
========================================================================

  Running from: /home/YOUR_USERNAME/YOUR_DIRECTORY

  Setup wizard running at:
  👉 http://localhost:8000
  👉 http://0.0.0.0:8000

  Press CTRL+C to stop
========================================================================
```

### Access Options:

**Same machine?**
- http://localhost:8000

**Different machine/Windows → Linux?**
- http://YOUR_SERVER_IP:8000
- Find your IP with: `hostname -I | awk '{print $1}'`

**Can't reach the server?**
- Make sure you're using `http://` NOT `https://`
- Try incognito/private mode in your browser
- Clear browser cache (Ctrl+F5)
- Check firewall: `sudo ufw allow 8000` (if using ufw)

---

## ✨ WHAT'S FIXED

### Master Code Optimizations:

1. **Auto-Path Detection**
   - Detects installation directory automatically
   - Works from `/home/user`, `/home/croft`, or any location
   - No more hardcoded paths!

2. **Universal Compatibility**
   - Scans multiple locations for avatar system
   - Adapts to your directory structure
   - Works regardless of where you cloned the repo

3. **Embedded HTML**
   - All HTML is embedded in Python
   - No external file dependencies
   - No more FileNotFoundError!

4. **Smart Error Handling**
   - Graceful fallbacks for missing scripts
   - Clear error messages with solutions
   - Real-time status updates

5. **Enhanced UI**
   - Shows detected paths in wizard
   - Real-time installation logs
   - Better feedback and progress tracking

---

## 📋 WIZARD FEATURES

### Step 1: System Status
- ✅ Checks Avatar System installation
- ✅ Checks T-FAN Cockpit
- ✅ Checks Ollama AI
- ✅ Checks Dependencies (Python, FFmpeg, Git)
- 📁 **Shows detected paths**

### Step 2: Install Components
- **One-click Install** for:
  - System Dependencies (FFmpeg, Ollama, etc.)
  - T-FAN Cockpit (downloads from GitHub)
  - WebKit Fixes (compatibility patches)
- **Real-time logs** show installation progress
- **Auto-refresh** status after installation

### Step 3: Launch
- 🎤 **Voice Interface** - Talk to Ara
- 🌐 **Web Chat** - Multi-AI interface
- 🖥️ **T-FAN Cockpit** - Spaceship HUD
- 🎬 **Avatar API** - Generate talking videos

---

## 🔧 TROUBLESHOOTING

### "Server can't be reached"

1. **Check server is running:**
   ```bash
   ps aux | grep setup_server
   ```

2. **Check port 8000 is listening:**
   ```bash
   sudo lsof -i :8000
   ```

3. **Restart the server:**
   ```bash
   pkill -f setup_server.py
   python3 setup_server.py
   ```

4. **Try different port (if 8000 is blocked):**
   Edit `setup_server.py`, change last line:
   ```python
   uvicorn.run(app, host="0.0.0.0", port=8080)  # Use 8080 instead
   ```

### "ModuleNotFoundError: fastapi"

```bash
sudo apt install -y python3-fastapi python3-uvicorn
```

Or if that fails:
```bash
pip3 install fastapi uvicorn --break-system-packages
```

### Running from wrong directory?

The new setup_server.py works from **ANYWHERE**! Just run:
```bash
python3 /path/to/setup_server.py
```

It will auto-detect everything!

---

## 🎯 DIRECT API ACCESS

If you prefer command-line over web UI:

### Check Status
```bash
curl http://localhost:8000/api/system/status | python3 -m json.tool
```

### Install T-FAN
```bash
curl -X POST http://localhost:8000/api/system/install-tfan
```

### Fix WebKit
```bash
curl -X POST http://localhost:8000/api/system/fix-webkit
```

### Launch Components
```bash
curl -X POST http://localhost:8000/api/system/launch/voice
curl -X POST http://localhost:8000/api/system/launch/tfan
curl -X POST http://localhost:8000/api/system/launch/api
```

---

## 📞 SUPPORT

**Issue?** Check the console output where you ran `setup_server.py` - it shows detailed error messages.

**Still stuck?** The server logs show exactly what's happening:
- Green text = Success
- Red text = Error (read the message for solution)
- Yellow text = Warning (usually safe to ignore)

---

## 🚀 NEXT STEPS

1. **Run the setup server** (choose any option above)
2. **Open http://localhost:8000** in your browser
3. **Follow the 3-step wizard:**
   - Step 1: Check what's installed
   - Step 2: Install missing components
   - Step 3: Launch your avatar system!

---

**YOU ARE NOW READY TO USE THE ARA AVATAR SYSTEM!**

The code is debugged, optimized, and works perfectly from any location.

**Go to:** http://localhost:8000
