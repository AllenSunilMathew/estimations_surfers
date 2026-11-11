"""Game engine — player, obstacles, coins, collision detection."""

import math
import random

from constants import (
    Gesture,
    INITIAL_SPEED,
    SPEED_INCREMENT,
    SPEED_INCREMENT_INTERVAL,
    MAX_SPEED,
    OBSTACLE_SPAWN_MIN,
    OBSTACLE_SPAWN_MAX,
    COIN_SPAWN_INTERVAL,
    INITIAL_LIVES,
    COIN_SCORE,
    DISTANCE_SCORE_RATE,
    PLAYER_JUMP_DURATION,
    PLAYER_SLIDE_DURATION,
    PLAYER_LANE_SWITCH_SPEED,
    BIRD_SPEED_FACTOR,
)


# ────────────────────────────────────────────────────────────────────
# Player
# ────────────────────────────────────────────────────────────────────
class Player:
    def __init__(self):
        self.lane = 1           # 0 left · 1 centre · 2 right
        self.target_lane = 1
        self.lane_progress = 0.0
        self.y_offset = 0.0     # vertical offset for jump arc
        self.is_jumping = False
        self.is_sliding = False
        self.jump_timer = 0.0
        self.slide_timer = 0.0
        self.can_act = True

    def update(self, dt):
        # Jump arc
        if self.is_jumping:
            self.jump_timer += dt
            t = self.jump_timer / PLAYER_JUMP_DURATION
            if t >= 1.0:
                self.is_jumping = False
                self.jump_timer = 0.0
                self.y_offset = 0.0
                self.can_act = True
            else:
                self.y_offset = -math.sin(t * math.pi) * 120

        # Slide
        if self.is_sliding:
            self.slide_timer += dt
            if self.slide_timer >= PLAYER_SLIDE_DURATION:
                self.is_sliding = False
                self.slide_timer = 0.0
                self.can_act = True

        # Smooth lane switch
        if self.lane != self.target_lane:
            self.lane_progress += dt / PLAYER_LANE_SWITCH_SPEED
            if self.lane_progress >= 1.0:
                self.lane = self.target_lane
                self.lane_progress = 0.0

    # ── actions ─────────────────────────────────────────────────────

    def jump(self):
        if self.can_act and not self.is_jumping and not self.is_sliding:
            self.is_jumping = True
            self.jump_timer = 0.0
            self.can_act = False

    def slide(self):
        if self.can_act and not self.is_jumping and not self.is_sliding:
            self.is_sliding = True
            self.slide_timer = 0.0
            self.can_act = False

    def move_left(self):
        if self.target_lane > 0 and self.lane == self.target_lane:
            self.target_lane -= 1
            self.lane_progress = 0.0

    def move_right(self):
        if self.target_lane < 2 and self.lane == self.target_lane:
            self.target_lane += 1
            self.lane_progress = 0.0

    def steer(self, desired_lane):
        """Head one lane at a time towards the lane the body is in."""
        if desired_lane is None or self.lane != self.target_lane:
            return
        if desired_lane < self.target_lane:
            self.move_left()
        elif desired_lane > self.target_lane:
            self.move_right()

    def collision_lane(self):
        """Lane used for hit tests: switches half-way through a lane change."""
        if self.lane != self.target_lane and self.lane_progress >= 0.5:
            return self.target_lane
        return self.lane

    def visual_lane(self):
        """Interpolated lane for smooth rendering."""
        if self.lane != self.target_lane:
            return self.lane + (self.target_lane - self.lane) * self.lane_progress
        return float(self.lane)


# ────────────────────────────────────────────────────────────────────
# Obstacle / Coin
# ────────────────────────────────────────────────────────────────────
class Obstacle:
    def __init__(self, lane: int, obs_type: str):
        self.lane = lane
        self.type = obs_type      # "barrier" | "low_bar" | "bird"
        self.z = 0.0              # 0 = horizon, 1 = at player
        self.active = True
        self.phase = random.uniform(0, 6.28)   # wing-flap / sway offset (birds)

    def update(self, speed, dt):
        k = BIRD_SPEED_FACTOR if self.type == "bird" else 1.0
        self.z += speed * dt * 0.14 * k
        if self.z > 1.25:
            self.active = False


class Coin:
    def __init__(self, lane: int):
        self.lane = lane
        self.z = 0.0
        self.collected = False
        self.active = True
        self.rotation = 0.0

    def update(self, speed, dt):
        self.z += speed * dt * 0.14
        self.rotation += dt * 300
        if self.z > 1.25:
            self.active = False


# ────────────────────────────────────────────────────────────────────
# Main game engine
# ────────────────────────────────────────────────────────────────────
class GameEngine:
    def __init__(self):
        self.player = Player()
        self.obstacles: list[Obstacle] = []
        self.coins: list[Coin] = []
        self.coin_collect_events: list[tuple[int, float]] = []
        self.hit_events: list[tuple[int, float, str]] = []
        self.score = 0
        self.lives = INITIAL_LIVES
        self.speed = INITIAL_SPEED
        self.game_over = False
        self.distance = 0.0
        self.flash_timer = 0.0
        self._prev_gesture = Gesture.NEUTRAL
        self._obstacle_timer = 0.0
        self._coin_timer = 0.0
        self._next_obstacle_interval = self._rand_obstacle_interval()

    def reset(self):
        self.__init__()

    # ── tick ────────────────────────────────────────────────────────

    def update(self, gesture: str, dt: float, lane_target=None):
        if self.game_over:
            return

        self.coin_collect_events.clear()
        self.hit_events.clear()

        # Edge‑triggered gesture → action
        if gesture != self._prev_gesture:
            if gesture == Gesture.JUMP:
                self.player.jump()
            elif gesture == Gesture.SLIDE:
                self.player.slide()
        self._prev_gesture = gesture

        # Body position -> lane (step left = run left, step right = run right)
        self.player.steer(lane_target)

        self.player.update(dt)

        # Speed ramp
        level = self.score // SPEED_INCREMENT_INTERVAL
        self.speed = min(INITIAL_SPEED + level * SPEED_INCREMENT, MAX_SPEED)

        # Spawn obstacles
        self._obstacle_timer += dt
        if self._obstacle_timer >= self._next_obstacle_interval:
            self._obstacle_timer = 0.0
            self._next_obstacle_interval = self._rand_obstacle_interval()
            lane = random.randint(0, 2)
            t = random.choice(["barrier", "barrier", "low_bar", "bird", "bird"])
            self.obstacles.append(Obstacle(lane, t))

        # Spawn coins
        self._coin_timer += dt
        if self._coin_timer >= COIN_SPAWN_INTERVAL:
            self._coin_timer = 0.0
            lane = random.randint(0, 2)
            if not any(o.lane == lane and o.z < 0.25 for o in self.obstacles):
                self.coins.append(Coin(lane))

        # Update world
        for o in self.obstacles:
            o.update(self.speed, dt)
        self.obstacles = [o for o in self.obstacles if o.active]

        for c in self.coins:
            c.update(self.speed, dt)
        self.coins = [c for c in self.coins if c.active]

        if self.flash_timer > 0:
            self.flash_timer -= dt

        self._check_collisions()

        self.distance += self.speed * dt
        self.score = int(self.distance * DISTANCE_SCORE_RATE)

    # ── helpers ─────────────────────────────────────────────────────

    def _rand_obstacle_interval(self):
        return random.uniform(OBSTACLE_SPAWN_MIN, OBSTACLE_SPAWN_MAX)

    def _check_collisions(self):
        plr = self.player.collision_lane()

        for o in self.obstacles:
            if not o.active:
                continue
            if 0.82 <= o.z <= 1.0 and o.lane == plr:
                if o.type == "barrier" and self.player.is_jumping:
                    continue
                if o.type in ("low_bar", "bird") and self.player.is_sliding:
                    continue          # duck under bars and birds
                # hit!
                o.active = False
                self.hit_events.append((o.lane, o.z, o.type))
                self.lives -= 1
                self.flash_timer = 0.30
                if self.lives <= 0:
                    self.game_over = True

        for c in self.coins:
            if c.collected or not c.active:
                continue
            if 0.80 <= c.z <= 1.0 and c.lane == plr:
                c.collected = True
                c.active = False
                self.score += COIN_SCORE
                self.coin_collect_events.append((c.lane, c.z))