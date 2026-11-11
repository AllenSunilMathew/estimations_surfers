# 🏆 Treasure Runner — AI Pose Controlled Game

An endless runner game controlled entirely by your body using AI pose estimation.

## Requirements

- Python 3.9+
- A working webcam

## Installation

```bash
cd treasure_runner
pip install -r requirements.txt
```

## Running the Game

```bash
python main.py
```

## Controls

| Body Pose | Game Action |
|---|---|
| ⬅️ Step / lean **left** of your centre mark | Character runs to the **left** lane |
| ➡️ Step / lean **right** of your centre mark | Character runs to the **right** lane |
| ⬆️ Both arms raised | **Jump** over barriers |
| ⬇️ Sit / crouch down | **Slide** under low bars and **duck birds** |

Stand still for the calibration bar: that spot becomes your centre. Two dashed
white lines then appear on the camera feed - move past the left line for the
left lane, past the right line for the right lane, and come back between them
for the middle lane.

## Obstacles

- 🚧 **Barrier** – jump over it (or change lane)
- 🟨 **Low bar** – slide under it (or change lane)
- 🐦 **Bird** – flies fast at head height; crouch to duck under it (jumping will hit it!) or change lane

You can also press **SPACE** to start the built-in game and **ESC** to quit/return to menu.

## Music

Put two files in an `assets` folder next to `main.py`:

- `bgm.mp3` — loading screen, menu and gameplay (loops)
- `gameover.mp3` — game-over screen (plays once)

`.ogg` / `.wav` also work, and files with "over" in the name are used as the
game-over track. Press **M** to mute / unmute. Use music you have the rights to use.

## Tips

- Stand far enough from the camera so your full upper body is visible.
- Keep a neutral standing pose for calibration during the first few seconds.
- Exaggerate your gestures for better detection.
- Good lighting helps pose detection accuracy.# pose-estimation-game