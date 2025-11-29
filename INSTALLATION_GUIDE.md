# Ara Installation Guide

## Two Installation Options

### Option 1: Ara Only (Lightweight)
**What you get:**
- ✅ AI co-pilot with Ara persona
- ✅ Voice control with 40+ macros
- ✅ Talking avatar generation
- ✅ Offline chat with Ollama
- ✅ Multi-AI delegation
- ❌ No T-FAN cockpit HUD
- ❌ No metrics visualization

**Install:**
```bash
./setup_ara.sh
# Choose option 1
```

**Use cases:**
- Just want the AI assistant
- Don't need system monitoring
- Lighter on resources
- Faster installation

---

### Option 2: Ara + T-FAN (Complete System)
**What you get:**
- ✅ Everything from Option 1
- ✅ T-FAN spaceship-style cockpit HUD
- ✅ GPU, CPU, network, storage metrics
- ✅ Topology visualization
- ✅ Workspace modes (work/relax/focus)
- ✅ Full voice macro integration with cockpit

**Install:**
```bash
./setup_ara.sh
# Choose option 2
```

**Use cases:**
- Want the full experience
- Need system metrics monitoring
- Want the sci-fi cockpit aesthetic
- Using voice macros like "show gpu", "red alert", etc.

---

## Quick Decision Guide

**Choose Ara Only if:**
- You just want to chat with Ara
- You don't care about system metrics
- You want faster installation
- You're on limited hardware

**Choose Ara + T-FAN if:**
- You want the complete co-pilot experience
- You want to monitor system metrics
- You love sci-fi themed interfaces
- You want to use commands like "red alert", "warp drive", "shields up"

---

## Can I add T-FAN later?

**Yes!** If you start with "Ara Only", you can add T-FAN anytime:

```bash
./install_complete_system.sh
```

This will add the cockpit to your existing Ara installation without breaking anything.

---

## Installation Scripts Reference

| Script | Purpose |
|--------|---------|
| `setup_ara.sh` | Interactive setup (choose Ara only or Ara+T-FAN) |
| `install_complete_system.sh` | Direct install of Ara + T-FAN |
| `start_ara.sh` | Launcher menu after installation |

---

## After Installation

Launch Ara with:
```bash
./start_ara.sh
```

You'll see a menu:
```
1) 🎙️  Voice Mode      - Voice-controlled Ara
2) 💬 Chat Mode       - Text chat
3) 🎬 Avatar Demo     - Generate talking avatar
4) 🚀 T-FAN Cockpit   - Launch cockpit (if installed)
5) 🌐 Multi-AI Server - Web interface
6) 📋 List Macros     - Show all commands
7) ⚙️  Settings        - Configure Ara
8) 🧪 System Check    - Test dependencies
```

---

## File Locations

**Ara files:**
- `ara_voice_interface.py` - Voice/chat interface
- `multi-ai-workspace/` - Multi-AI system
- `assets/avatars/` - Avatar images
- `ARA_README.md` - Full documentation

**T-FAN files** (if installed):
- `~/tfan-ara-system/Quanta-meis-nib-cis/` - T-FAN cockpit
- Installed by `install_complete_system.sh`

---

## Need Help?

1. Read `ARA_README.md` for full documentation
2. Run system check: `./start_ara.sh` → option 8
3. Check Ollama is running: `ollama serve`

---

**Recommended:** Start with Option 1 (Ara Only) to test everything, then add T-FAN later if you want the full cockpit experience.
