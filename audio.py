"""Music for Treasure Runner — one track for gameplay, one for game over.

Put both files in an `assets` folder next to main.py. Easiest names:
    bgm.mp3        -> plays on the loading screen, menu and while playing (loops)
    gameover.mp3   -> plays on the game-over screen (plays once)
(.mp3 / .ogg / .wav all work.)

Other names also work: any file with "over", "lose" or "fail" in its name is
used as the game-over track, and any other audio file as the gameplay track.
Or set GAMEPLAY_FILE / GAMEOVER_FILE below to exact file names.

Press M in the game to mute / unmute. Missing files = that part stays silent.
"""

import os

import pygame

GAMEPLAY_FILE = None        # e.g. "my_song.mp3"  (None = auto-detect)
GAMEOVER_FILE = None        # e.g. "sad_trombone.mp3"
GAMEPLAY_VOLUME = 0.5       # 0.0 – 1.0
GAMEOVER_VOLUME = 0.7
GAMEOVER_LOOP = False       # True = repeat the game-over track until restart

_EXT = (".mp3", ".ogg", ".wav")
_OVER_WORDS = ("over", "lose", "lost", "fail", "dead", "death")
_PLAY_WORDS = ("bgm", "game", "play", "music", "theme", "loop")


class MusicPlayer:
    def __init__(self):
        self.ok = False
        self.muted = False
        self._scene = None
        self.tracks = {"game": None, "over": None}
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            self.ok = True
        except Exception as e:
            print(f"[WARN] Audio disabled ({e})")
            return
        self.tracks = self._find_tracks()
        for k, v in self.tracks.items():
            label = "gameplay" if k == "game" else "game-over"
            print(f"[INFO] {label} music: {os.path.basename(v) if v else 'NOT FOUND'}")

    # ── file discovery ──────────────────────────────────────────────
    @staticmethod
    def _find_tracks():
        base = os.path.dirname(os.path.abspath(__file__))
        files = []
        for folder in (os.path.join(base, "assets"), base):
            if os.path.isdir(folder):
                for n in sorted(os.listdir(folder)):
                    if n.lower().endswith(_EXT):
                        files.append(os.path.join(folder, n))

        def exact(name):
            if not name:
                return None
            for f in files:
                if os.path.basename(f).lower() == name.lower():
                    return f
            return None

        over = exact("./assets/game.mp3")
        game = exact("./assets/subway_surfers.mp3")
        if over is None:
            over = next((f for f in files if any(w in os.path.basename(f).lower() for w in _OVER_WORDS)), None)
        rest = [f for f in files if f != over]
        if game is None:
            game = next((f for f in rest if any(w in os.path.basename(f).lower() for w in _PLAY_WORDS)), None)
            if game is None and rest:
                game = rest[0]
        return {"game": game, "over": over}

    # ── playback ────────────────────────────────────────────────────
    def _volume(self):
        if self.muted:
            return 0.0
        return GAMEOVER_VOLUME if self._scene == "over" else GAMEPLAY_VOLUME

    def set_scene(self, scene):
        """scene: 'game' or 'over'. Call every frame; it only acts when the scene changes."""
        if not self.ok or scene == self._scene:
            return
        self._scene = scene
        path = self.tracks.get(scene)
        try:
            pygame.mixer.music.stop()
            if path:
                pygame.mixer.music.load(path)
                loops = -1 if (scene == "game" or GAMEOVER_LOOP) else 0
                pygame.mixer.music.play(loops, fade_ms=250)
            pygame.mixer.music.set_volume(self._volume())
        except Exception as e:
            print(f"[WARN] Could not play {path}: {e}")

    def toggle_mute(self):
        self.muted = not self.muted
        if self.ok:
            pygame.mixer.music.set_volume(self._volume())

    def stop(self):
        if self.ok:
            pygame.mixer.music.stop()
