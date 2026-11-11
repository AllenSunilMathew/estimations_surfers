"""Audio manager module for Treasure Runner game.

Handles background music, hit sounds, and game over sounds using pygame.mixer.
Gracefully handles missing audio files and mixer failures.
"""

import os
import pygame


class AudioManager:
    """Manages all audio playback for the Treasure Runner game."""

    def __init__(self):
        """Initialize pygame.mixer and load all sound assets.

        Uses relative paths from this file's directory to locate audio files
        in the 'assests' folder. Missing files produce a warning but do not
        crash the game.
        """
        self.sounds_loaded = False
        self.hit_sound = None
        self.game_over_sound = None

        try:
            pygame.mixer.init()
        except pygame.error as e:
            print(f"[AudioManager] Warning: Could not initialize mixer: {e}")
            return

        assets_dir = os.path.join(os.path.dirname(__file__), "assests")

        # --- Background music (streamed via pygame.mixer.music) ---
        music_path = os.path.join(assets_dir, "subway_surfers.mp3")
        if os.path.isfile(music_path):
            try:
                pygame.mixer.music.load(music_path)
                pygame.mixer.music.set_volume(0.4)
            except pygame.error as e:
                print(f"[AudioManager] Warning: Could not load background music: {e}")
        else:
            print(f"[AudioManager] Warning: Background music not found at {music_path}")

        # --- Hit sound effect ---
        hit_path = os.path.join(assets_dir, "hit.mp3")
        if os.path.isfile(hit_path):
            try:
                self.hit_sound = pygame.mixer.Sound(hit_path)
                self.hit_sound.set_volume(0.7)
            except pygame.error as e:
                print(f"[AudioManager] Warning: Could not load hit sound: {e}")
        else:
            print(f"[AudioManager] Warning: Hit sound not found at {hit_path}")

        # --- Game over sound effect ---
        game_over_path = os.path.join(assets_dir, "game_over.mp3")
        if os.path.isfile(game_over_path):
            try:
                self.game_over_sound = pygame.mixer.Sound(game_over_path)
                self.game_over_sound.set_volume(0.8)
            except pygame.error as e:
                print(f"[AudioManager] Warning: Could not load game over sound: {e}")
        else:
            print(f"[AudioManager] Warning: Game over sound not found at {game_over_path}")

        self.sounds_loaded = True

    def play_music(self):
        """Start background music, looping indefinitely."""
        try:
            pygame.mixer.music.play(-1)
        except pygame.error as e:
            print(f"[AudioManager] Warning: Could not play music: {e}")

    def stop_music(self):
        """Stop background music with a 500ms fadeout."""
        try:
            pygame.mixer.music.fadeout(500)
        except pygame.error as e:
            print(f"[AudioManager] Warning: Could not stop music: {e}")

    def play_hit(self):
        """Play the hit sound effect (only if loaded)."""
        if self.hit_sound:
            try:
                self.hit_sound.play()
            except pygame.error as e:
                print(f"[AudioManager] Warning: Could not play hit sound: {e}")

    def play_game_over(self):
        """Stop background music, then play the game over sound (only if loaded)."""
        self.stop_music()
        if self.game_over_sound:
            try:
                self.game_over_sound.play()
            except pygame.error as e:
                print(f"[AudioManager] Warning: Could not play game over sound: {e}")

    def pause_music(self):
        """Pause background music (e.g. for menu transitions)."""
        try:
            pygame.mixer.music.pause()
        except pygame.error as e:
            print(f"[AudioManager] Warning: Could not pause music: {e}")

    def unpause_music(self):
        """Unpause background music."""
        try:
            pygame.mixer.music.unpause()
        except pygame.error as e:
            print(f"[AudioManager] Warning: Could not unpause music: {e}")
