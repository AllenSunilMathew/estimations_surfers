"""Constants and configuration for Treasure Runner game."""


# ── Window ──────────────────────────────────────────────────────────
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
FPS_TARGET = 30
GAME_TITLE = "Treasure Runner — AI Pose Controlled Game"

# ── Camera ──────────────────────────────────────────────────────────
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# ── Layout ──────────────────────────────────────────────────────────
HUD_HEIGHT = 100
POSE_PANEL_RATIO = 0.46  # Left panel width ratio

# ── Colors — "sky card" theme (same look as the preloader) ─────────
# Page + card
PAGE_BG = (236, 241, 246)          # off-white page behind the card
SKY_A = (122, 214, 234)            # card gradient: centre
SKY_B = (82, 182, 221)             # card gradient: edges
# Panels drawn on top of the card
PANEL_FILL = (70, 170, 212)
PANEL_DARK = (36, 128, 176)        # header bands / dark pills
ROW_FILL = (52, 150, 196)
EDGE_SOFT = (190, 236, 248)
# Text
INK = (30, 110, 150)               # deep teal (text on white)
INK_SHADOW = (40, 120, 160)        # drop-shadow colour for white text
SOFT_WHITE = (232, 246, 252)
# Accents (taken from the runner + coin artwork)
GOLD = (251, 191, 36)
GOLD_BRIGHT = (255, 200, 40)
HOODIE_RED = (214, 50, 52)
JEANS_BLUE = (52, 98, 172)
GREEN = (34, 197, 94)
GREEN_DARK = (30, 160, 90)
ORANGE = (249, 115, 22)
ORANGE_DARK = (220, 120, 20)
RED = (239, 68, 68)
BLUE_DARK = (40, 110, 200)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# ── Gesture thresholds (normalised 0‑1 coords) ────────────────────
GESTURE_ARM_THRESHOLD = 0.05      # Wrist above shoulder
GESTURE_DIAGONAL_THRESHOLD = 0.08  # Wrist‑Y diff for / and \
GESTURE_CROUCH_THRESHOLD = 0.12   # Nose‑drop for squat
CALIBRATION_FRAMES = 30
LANE_MOVE_THRESHOLD = 0.08        # body sideways shift (fraction of frame width) to change lane

# ── Game mechanics ──────────────────────────────────────────────────
INITIAL_SPEED = 4.5
SPEED_INCREMENT = 0.4
SPEED_INCREMENT_INTERVAL = 500     # points between speed bumps
MAX_SPEED = 14.0
OBSTACLE_SPAWN_MIN = 0.9          # seconds between obstacles
OBSTACLE_SPAWN_MAX = 1.8
COIN_SPAWN_INTERVAL = 0.7
INITIAL_LIVES = 3
COIN_SCORE = 50
BIRD_SPEED_FACTOR = 1.3            # birds fly faster than the ground scrolls
DISTANCE_SCORE_RATE = 2            # points per unit‑distance

# ── Player animation ───────────────────────────────────────────────
PLAYER_JUMP_DURATION = 0.55
PLAYER_SLIDE_DURATION = 0.55
PLAYER_LANE_SWITCH_SPEED = 0.15    # seconds for smooth lane slide


# ── Enums ───────────────────────────────────────────────────────────
class Gesture:
    NEUTRAL = "Neutral"
    JUMP = "Jump"
    SLIDE = "Slide"
    LEFT = "Left"
    RIGHT = "Right"


class State:
    PRELOADER = "preloader"
    MENU = "menu"
    PLAYING = "playing"
    GAME_OVER = "game_over"