import pygame
import pygame.freetype
import sys
import cv2
import math
import dlib
import numpy as np
from threading import Thread
from queue import Queue
from imutils.video import VideoStream
import imutils
from face_utils import (
    get_mouth_loc_with_height,
    draw_mouth,
    get_head_tilt_angle,
    shape_to_np,
    mouth_aspect_ratio,
    MOUTH_AR_THRESH,
)
from detect_tongue_tip_real_time import check_tongue_for_player
import detect_tongue_tip_real_time
import time

# Frame queue for thread-safe communication
frame_queue = Queue(maxsize=2)
detection_queue = Queue(maxsize=2)  # Queue for face angle data
tongue_queue = Queue(maxsize=2)  # Queue for tongue states
running = True
last_frame = None

# Initialize dlib face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def camera_capture_thread():
    """Captures frames from camera with AI processing"""
    global running
    vs = VideoStream(0).start()

    while running:
        frame = vs.read()
        frame = cv2.flip(frame, 1)  # flip horizontally
        if frame is None:
            continue

        enhanced = cv2.detailEnhance(frame, sigma_s=10, sigma_r=0.15)
        frame = imutils.resize(frame, width=300)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect all faces in frame
        rects = detector(gray, 1)

        # Extract angles and tongue states for each face (max 2)
        face_angles = []
        tongue_states = [False, False]

        frame_center_x = frame.shape[1] // 2  # Midpoint of frame width

        for face_idx, rect in enumerate(rects):
            if face_idx >= 2:  # Max 2 players
                break

            # Determine player_id based on face x-position (left/right)
            face_center_x = (rect.left() + rect.right()) // 2
            player_id = 0 if face_center_x < frame_center_x else 1

            # Get facial landmarks
            shape = predictor(gray, rect)
            shape = shape_to_np(shape)

            # Draw mouth landmarks on frame
            frame = draw_mouth(frame, shape)

            # Get head tilt angle and draw line from chin to nose
            angle, nose_top, chin_bottom = get_head_tilt_angle(shape)
            cv2.line(
                frame, tuple(nose_top), tuple(chin_bottom), (0, 255, 255), 2
            )  # Yellow line

            face_angles.append({"face_id": player_id, "angle": angle})

            # Check tongue for this player
            try:
                # Extract mouth location and dimensions
                from imutils import face_utils

                mouth_x, mouth_y, mouth_w, mouth_h = 0, 0, 0, 0
                inner_mouth_y = 0

                for name, (i, j) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                    if name == "mouth":
                        mouth_x, mouth_y, mouth_w, mouth_h = cv2.boundingRect(
                            np.array([shape[i:j]])
                        )
                    if name == "inner_mouth":
                        inner_x, inner_y, inner_w, inner_h = cv2.boundingRect(
                            np.array([shape[i:j]])
                        )
                        inner_mouth_y = inner_y

                mouth_data = {
                    "mouth_x": mouth_x,
                    "mouth_y": mouth_y,
                    "mouth_w": mouth_w,
                    "mouth_h": mouth_h,
                    "inner_mouth_y": inner_mouth_y,
                }

                tongue_states[player_id] = check_tongue_for_player(
                    shape, enhanced, mouth_data
                )
            except:
                tongue_states[player_id] = False

        # Send face angles to main loop
        if face_angles and not detection_queue.full():
            detection_queue.put(face_angles)

        # Send tongue states to main loop
        if not tongue_queue.full():
            tongue_queue.put(tongue_states)

        # Convert BGR to RGB for pygame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = frame_rgb.swapaxes(0, 1)

        if not frame_queue.full():
            frame_queue.put(frame_rgb)

    vs.stop()


def get_player_head_angles():
    """Get head tilt angles for both players (non-blocking)"""
    if not detection_queue.empty():
        return detection_queue.get()
    return None


def get_tongue_states():
    """Get tongue states for both players (non-blocking)"""
    if not tongue_queue.empty():
        return tongue_queue.get()
    return [False, False]


pygame.init()

myFont = pygame.freetype.Font("assets/papyrus.ttf", 36)

gridWidth = 800
gridHeight = 600

charWidth = 200
charHeight = 200
tongueMaxWidth = 600

screen = pygame.display.set_mode((gridWidth, gridHeight))

_frog_img = cv2.imread("assets/frog.png", cv2.IMREAD_UNCHANGED)
_frog_img = cv2.resize(
    _frog_img, (charWidth, charHeight), interpolation=cv2.INTER_NEAREST
)
if _frog_img.shape[2] == 4:
    _frog_img = cv2.cvtColor(_frog_img, cv2.COLOR_BGRA2RGBA)
    frog_sprite = pygame.Surface((charWidth, charHeight), pygame.SRCALPHA)
    _t = _frog_img.swapaxes(0, 1)
    pygame.surfarray.pixels3d(frog_sprite)[:] = _t[:, :, :3]
    pygame.surfarray.pixels_alpha(frog_sprite)[:] = _t[:, :, 3]
else:
    _frog_img = cv2.cvtColor(_frog_img, cv2.COLOR_BGR2RGB)
    frog_sprite = pygame.surfarray.make_surface(_frog_img.swapaxes(0, 1))
frog_sprite_flipped = pygame.transform.flip(frog_sprite, True, False)
clock = pygame.time.Clock()

# Start camera thread
cam_thread = Thread(target=camera_capture_thread, daemon=True)
cam_thread.start()


class player:
    def __init__(self, x, y, player_id):
        self.x = x
        self.y = y
        self.health = 100
        self.player_id = player_id
        self.state = 0  # -1 0 1 for left, neutral, right
        self.tongue_active = False
        self.tongue_width = 0
        self.tongue_hit = False

    def move(self, dx):
        self.x += dx
        if self.state > -1 and dx < 0:
            self.state -= 1
        elif self.state < 1 and dx > 0:
            self.state += 1

    @property
    def hitbox(self):
        inset_x = charWidth // 5
        inset_top = charHeight // 3
        inset_bottom = 0
        return pygame.Rect(
            self.x + inset_x,
            self.y + inset_top,
            charWidth - inset_x * 2,
            charHeight - inset_top - inset_bottom,
        )

    def attack(self):
        self.tongue_active = True
        self.tongue_width = 0
        self.tongue_hit = False

    @property
    def tongue_rect(self):
        tongue_height = 20
        y = self.hitbox.centery - tongue_height // 2
        if self.player_id == 0:
            return pygame.Rect(self.hitbox.right, y, self.tongue_width, tongue_height)
        else:
            return pygame.Rect(
                self.hitbox.left - self.tongue_width,
                y,
                self.tongue_width,
                tongue_height,
            )

    def update_tongue(self, other_player):
        if not self.tongue_active:
            return
        self.tongue_width += 50
        if self.tongue_width >= tongueMaxWidth:
            self.tongue_width = tongueMaxWidth
            self.tongue_active = False
        if not self.tongue_hit and self.tongue_rect.colliderect(other_player.hitbox):
            print("Successful attack")
            self.tongue_hit = True

    def draw_tongue(self, screen):
        if self.tongue_active:
            pygame.draw.rect(screen, (255, 105, 180), self.tongue_rect)

    def draw(self, screen):
        sprite = frog_sprite_flipped if self.player_id == 1 else frog_sprite
        screen.blit(sprite, (self.x, self.y))

    def draw_hitbox(self, screen):
        pygame.draw.rect(screen, (255, 0, 0), self.hitbox, 2)

    def hurt(self):
        self.health -= 10
        # print(f"Player {self.player_id} hurt! Health: {self.health}")

    def draw_health_bar(self, screen):
        bar_width = 150
        bar_height = 20
        fill_width = int(bar_width * (self.health / 100))
        pygame.draw.rect(
            screen, (255, 0, 0), (self.x - 50, 25, bar_width, bar_height)
        )  # Red background
        pygame.draw.rect(
            screen, (0, 255, 0), (self.x - 50, 25, fill_width, bar_height)
        )  # Green health


# Player positions and Initialization
player1 = player((150 - (charWidth // 2)), gridHeight - charHeight - 100, 0)
player2 = player(gridWidth - (150 + (charWidth // 2)), gridHeight - charHeight - 100, 1)
players = [player1, player2]

running = True
prev = time.time()
while running:
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    angles = get_player_head_angles()
    if angles:
        for player_data in angles:
            player_id = player_data["face_id"]
            angle = player_data["angle"]
            # do something with player 0 or 1 and their angle

            # left=positive, right=negative relative to y axis
            if players[player_id].state > -1 and angle > 20:  # Player tilt left
                players[player_id].move(-50)  # Move left
                print(f"Player {player_id + 1} tilt left! Move left triggered.")
            elif players[player_id].state < 1 and angle < -20:  # Player tilt right
                players[player_id].move(50)  # Move right
                print(f"Player {player_id + 1} tilt right! Move right triggered.")

    tongue_states = get_tongue_states()
    # print(f"Tongue states: {tongue_states}")  # Debugging: print tongue states
    if tongue_states:
        for tongue_state in tongue_states:
            player_id = tongue_states.index(tongue_state)
            if tongue_state:  # Tongue is out
                cur = time.time()
                if cur - prev > 1:  # Prevent multiple attacks in quick succession
                    players[player_id].attack()  # Attack action
                    print(f"Player {player_id + 1} tongue out! Attack triggered.")
                    prev = cur

    screen.fill((255, 255, 255))

    myFont.render_to(screen, (100, 100), "testing, testing", (0, 0, 0))

    # Display camera frame with AI
    if not frame_queue.empty():
        last_frame = frame_queue.get()

    if last_frame is not None:
        surf = pygame.surfarray.make_surface(last_frame)
        screen.blit(surf, (250, 10))  # camera feed centered

    # Update and draw players
    for i, p in enumerate(players):
        p.update_tongue(players[1 - i])
    for p in players:
        p.draw(screen)
        p.draw_tongue(screen)
        p.draw_health_bar(screen)
        p.draw_hitbox(screen)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
