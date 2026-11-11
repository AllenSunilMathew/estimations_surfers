# 🏆 Treasure Runner — AI Pose Controlled 3D Game

An endless 3D runner game inspired by *Subway Surfers*, controlled entirely by your body movements and gestures using OpenCV and MediaPipe AI pose estimation in Python.

---

## 📋 Table of Contents

- [Features](#-features)
- [Pose Gestures & Controls](#-pose-gestures--controls)
- [Gameplay Mechanics](#-gameplay-mechanics)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Audio & Assets](#-audio--assets)
- [Architecture & File Structure](#-architecture--file-structure)

---

## ✨ Features

1. **AI Pose Estimation & Edge-Triggered Gesture Controls**:
   - Real-time 33-landmark pose tracking via MediaPipe Pose & OpenCV.
   - Robust normalization relative to shoulder/hip geometry with auto-calibration.

2. **Subway Surfers Characters: Police Inspector & Police Dog**:
   - **Game Start Chase**: The Police Inspector in navy uniform, peaked cap, and gold badge, along with his snarling, galloping Police Dog (with spiked red collar and panting tongue), chase close behind the runner at the start of each run before retreating.
   - **Stumble & Pursuit**: A small hit into a barrier or bar causes the runner to stumble, triggering an immediate high-speed pursuit (`🚨 PURSUIT: 5.5s`).
   - **Busted Fatal Hit**: Colliding while being pursued results in the Inspector tackling the runner for an instant Game Over (`CAUGHT BY THE INSPECTOR & DOG!`).

3. **Oncoming 3D Moving Subway Trains**:
   - 3D perspective subway carriages moving toward the runner at accelerated speeds (`1.45x`).
   - Modeled with twin glowing high-beam headlights projecting light onto the rails, driver windshield with glare, roof vents, and yellow-black hazard chevron cowcatchers.
   - Fatal obstacle: collision causes an instant fatal Hard Hit (`HIT BY AN ONCOMING TRAIN!`).

4. **Nexa Slab Typography**:
   - Full integration of the stylish **Nexa Slab** font family.
   - `Nexa Slab Bold` for display titles, scores, speed multipliers, and game-over banners.
   - `Nexa Slab Regular` for instructions, controls, live FPS/confidence readings, and button labels.

5. **Dynamic Speed Progression**:
   - Progressive speed scaling from `5.0` up to `16.0` as score increases.
   - Live `SPEED: X.Xx` badge with real-time motion speed lines.
   - Obstacle and coin spawn pacing dynamically adjusts to current speed.

6. **Threaded Non-Blocking Camera Manager**:
   - Multithreaded OpenCV capture engine utilizing `cv2.CAP_DSHOW` with dedicated background buffer handling.
   - Seamless, instant switching between Default Camera (`0`) and external Webcam (`1`) without UI freezing or "Not Responding" hangs.

7. **Native Window Management & Fullscreen**:
   - OS-native title bar Maximize / Restore button `[□]` enabled (`WS_MAXIMIZEBOX | WS_THICKFRAME`).
   - Canvas aspect-ratio scaling preserves sharp 1280×720 geometry at any window resolution or fullscreen mode (`F11`).

---

## 🏃 Pose Gestures & Controls

### Body Gestures (Tracked by Camera)

| Pose / Gesture | Visual Landmark Rule | In-Game Action |
|---|---|---|
| **Arms Up** | Wrists raised above eye level | **Jump** over barriers |
| **Sit / Crouch Down** | Hips drop significantly below baseline | **Slide / Duck** under low bars & birds |
| **`/` Shape** | Left hand down, Right hand up | **Move Left** lane |
| **`\` Shape** | Left hand up, Right hand down | **Move Right** lane |
| **Raise Both Hands** (At Game Over) | Both wrists raised | **Restart Game** |

### Keyboard & Mouse Shortcuts

| Key / Control | Action |
|---|---|
| **`C`** or Click **`📷 Swap [C]`** | Toggle between Camera 0 and Camera 1 |
| **`F11`** or Title Bar **`[□]`** | Maximize / Fullscreen mode |
| **`SPACE`** | Start run / Quick restart after Game Over |
| **`ESC`** | Return to Menu / Quit application |

---

## 🎮 Gameplay Mechanics

- **Single Life System**:
  - `INITIAL_LIVES = 1`.
  - **Small Hit / Stumble**: Hitting a barrier, low bar, or bird when safe causes a stumble animation, plays `hit.mp3`, and summons the Police Inspector & Dog for a 5.5s high-tension chase.
  - **Hard Hit / Fatal**:
    - Hitting an oncoming train is always fatal.
    - Hitting any obstacle while the police is actively pursuing close behind results in being caught (`game_over.mp3`).
- **Coin Collection**:
  - Rotating 3D gold coins generate bonus score and sparkle particles.

---

## 📦 Requirements

- **Operating System**: Windows 10 / 11, macOS, or Linux
- **Python**: 3.9, 3.10, or 3.11
- **Hardware**: Standard webcam or external USB camera

---

## 🚀 Installation

1. Open a terminal in the project directory:
   ```bash
   cd "C:\Users\Allen Sunil Mathew\.gemini\antigravity\scratch\treasure_runner"
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   *Dependencies in `requirements.txt`:*
   - `pygame>=2.5.0`
   - `opencv-python>=4.8.0`
   - `mediapipe>=0.10.0`
   - `numpy>=1.24.0`

---

## 🕹️ How to Run

Execute the main entry point:
```bash
python main.py
```

1. Stand in view of the camera so your upper body and arms are clearly visible.
2. The game will show a brief preloader and title screen with camera preview and controls guide.
3. Strike the **Arms Up** pose or press **`SPACE`** to start running!

---

## 🎵 Audio & Assets

Stored under `assests/` (and mirrored in `assets/`):
- `subway_surfers.mp3` — Fast-paced background theme music.
- `hit.mp3` — Impact sound for obstacle collisions and stumbles.
- `game_over.mp3` — Game over / busted audio jingle.
- `fonts/Nexa Slab Bold.otf` — High-impact slab serif font for titles & scores.
- `fonts/Nexa Slab Regular.otf` — Clear geometric slab serif font for UI & controls.

---

## 🏗️ Architecture & File Structure

```
treasure_runner/
│
├── main.py              # Application loop, window management, state machine
├── camera_manager.py    # Threaded non-blocking webcam capture with CAP_DSHOW
├── pose_detector.py     # MediaPipe pose estimation & gesture classifier
├── game_engine.py       # Game logic, player physics, moving trains, police pursuit, collisions
├── renderer.py          # 3D perspective projection, police & dog animation, train rendering, HUD
├── preloader.py         # Animated preloader splash screen & interactive main menu
├── theme.py             # Nexa Slab font resolver, sky card backgrounds, color palette
├── constants.py         # Timing, speed factors, window sizes, gesture definitions
├── requirements.txt     # Python package requirements
├── README.md            # Project documentation
│
└── assests/             # Audio and font resources
    ├── subway_surfers.mp3
    ├── hit.mp3
    ├── game_over.mp3
    └── fonts/
        ├── Nexa Slab Bold.otf
        └── Nexa Slab Regular.otf
```
# estimations_surfers
