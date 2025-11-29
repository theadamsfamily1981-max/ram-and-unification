# Ara-SYNERGY Wallpapers

This directory contains mode-specific wallpapers for the Ara cockpit interface.

## Required Wallpapers

### cruise_nebula.png
- **Mode**: CRUISE (IDLE state)
- **Aesthetic**: Calm, deep space nebula
- **Colors**: Deep blues, purples, blacks
- **Mood**: Peaceful, contemplative
- **Resolution**: 3840x2160 (4K) recommended

### flight_tactical.png
- **Mode**: FLIGHT (THINKING/PROCESSING state)
- **Aesthetic**: Tactical blue HUD overlay
- **Colors**: Cyan, electric blue, dark grays
- **Mood**: Alert, analytical, focused
- **Resolution**: 3840x2160 (4K) recommended

### battle_alert.png
- **Mode**: BATTLE (SPEAKING/CRITICAL state)
- **Aesthetic**: Red alert, critical status
- **Colors**: Deep reds, oranges, warning yellows
- **Mood**: Intense, urgent, high-energy
- **Resolution**: 3840x2160 (4K) recommended

## Installation

Place your wallpaper images in this directory with the exact filenames above.

The mode switcher script (`scripts/ara_mode.sh`) expects these files to exist at:
- `$HOME/wallpapers/cruise_nebula.png`
- `$HOME/wallpapers/flight_tactical.png`
- `$HOME/wallpapers/battle_alert.png`

## Creating Custom Wallpapers

### Design Guidelines

1. **Keep text minimal** - Let the visuals speak
2. **High contrast** - Ensure desktop icons remain visible
3. **Dark themes** - Reduce eye strain during long sessions
4. **Sci-fi aesthetic** - Match the Cathedral Avatar warship theme

### Recommended Tools

- **GIMP**: Free, open-source image editor
- **Blender**: 3D rendering for spacecraft/space scenes
- **AI Generation**: Stable Diffusion, Midjourney for concept art

### Example Prompts (for AI generation)

**CRUISE:**
```
Deep space nebula, purple and blue clouds, stars, peaceful,
high resolution, cinematic, photorealistic, 4K wallpaper
```

**FLIGHT:**
```
Tactical spaceship cockpit HUD, cyan holographic displays,
dark interface, futuristic, sci-fi, blue glow, 4K wallpaper
```

**BATTLE:**
```
Red alert spaceship interior, warning lights, critical systems,
intense red and orange lighting, sci-fi cockpit, 4K wallpaper
```

## Placeholder Usage

Until you create or download wallpapers, the system will attempt to use:
1. Files in this directory (if they exist)
2. Default GNOME wallpapers (fallback)

The mode switcher will still function, just without custom visuals.
