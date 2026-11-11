# 🏆 Treasure Runner — AI Pose Controlled Game

An endless runner game controlled entirely by your body movements using OpenCV and MediaPipe AI pose estimation.

## Requirements

- Python 3.9+
- A working webcam / camera

## Installation

```bash

pip install -r requirements.txt
```

## Running the Game

```bash
python main.py
```

## Controls & Keyboard Shortcuts

| Input | Action |
|---|---|
| **Body: Both arms raised** | **Jump** over barriers |
| **Body: Sit / Crouch down** | **Slide** under low bars & birds |
| **Body: Step / Lean left** | Steer character to **Left lane** |
| **Body: Step / Lean right** | Steer character to **Right lane** |
| **Click `⛶ FULLSCREEN` button** or press **`F11`** | **Toggle Fullscreen mode** (auto-detects screen resolution) |
| **Click `📷 Camera` button** or press **`C`** | **Instant Camera Swap** (switch between Camera 0 & Camera 1) |
| **`SPACE`** | Start game / Restart after Game Over |
| **`ESC`** | Return to Menu / Quit |

## Key Features

1. **Dynamic Speed Scaling**:
   - The game speed accelerates smoothly and noticeably as your score increases.
   - Live **SPEED multiplier** badge displayed on the HUD (e.g. `SPEED 1.2x`, `SPEED 2.0x ⚡`).
   - Obstacles and coins dynamically scale their arrival pacing to match the speed.
2. **Crash-Free Fullscreen with Screen Auto-Detection**:
   - Detects physical monitor resolution (`1536x864`, `1920x1080`, 2K, 4K).
   - High-performance virtual canvas rendering scales smoothly without Windows display driver crashes (`ChangeDisplaySettingsEx`).
   - Interactive on-screen button + `F11` hotkey.
3. **Non-Blocking Threaded Camera Capture**:
   - Uses a background worker thread (`CameraManager`) with `CAP_DSHOW` and zero buffer lag.
   - Switching between Default Camera (0) and Webcam (1) is instantaneous and never freezes or shows "Program Not Responding".
4. **Enhanced 3D Visuals & Audio**:
   - 3D running character with dynamic shading, highlights, motion trails, and jump/slide effects.
   - BGM (`subway_surfers.mp3`), collision SFX (`hit.mp3`), and death SFX (`game_over.mp3`).
