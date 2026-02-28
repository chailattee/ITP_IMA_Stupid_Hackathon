import pygame
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
)

# Frame queue for thread-safe communication
frame_queue = Queue(maxsize=2)
detection_queue = Queue(maxsize=2)  # Queue for face angle data
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

        # Extract angles for each face (max 2)
        face_angles = []
        for face_idx, rect in enumerate(rects):
            if face_idx >= 2:  # Max 2 players
                break

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

            face_angles.append({"face_id": face_idx, "angle": angle})

        # Send face angles to main loop
        if face_angles and not detection_queue.full():
            detection_queue.put(face_angles)

        # Convert BGR to RGB for pygame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = frame_rgb.swapaxes(0, 1)

        if not frame_queue.full():
            frame_queue.put(frame_rgb)

    vs.release()


def get_player_head_angles():
    """Get head tilt angles for both players (non-blocking)"""
    if not detection_queue.empty():
        return detection_queue.get()
    return None


pygame.init()
gridWidth = 800
gridHeight = 600

charWidth = 200
charHeight = 200

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
        self.player_id = player_id
        self.health = 100
        self.state = 0  # -1 0 1 for left, neutral, right

    def move(self, dx):
        self.x += dx
        if self.state > -1 and dx < 0:
            self.state -= 1
        elif self.state < 1 and dx > 0:
            self.state += 1

    def attack(self):
        print("Attack!")

    def draw(self, screen):
        sprite = frog_sprite_flipped if self.player_id == 1 else frog_sprite
        screen.blit(sprite, (self.x, self.y))

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


player1 = player((100 - (charWidth / 2)), gridHeight - charHeight, 0)
player2 = player(gridWidth - (100 + (charWidth / 2)), gridHeight - charHeight, 1)
players = [player1, player2]

running = True
while running:
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            pygame.quit()
            sys.exit()

    angles = get_player_head_angles()
    if angles:
        for player_data in angles:
            player_id = player_data["face_id"]
            angle = player_data["angle"]
            # do something with player 0 or 1 and their angle

            # left=positive, right=negative relative to y axis
            if players[player_id].state > -1 and angle > 20:  # Player tilt left
                players[player_id].move(-20)  # Move left
            elif players[player_id].state < 1 and angle < -20:  # Player tilt right
                players[player_id].move(20)  # Move right

    screen.fill((255, 255, 255))

    # Display camera frame with AI
    if not frame_queue.empty():
        last_frame = frame_queue.get()

    if last_frame is not None:
        surf = pygame.surfarray.make_surface(last_frame)
        screen.blit(surf, (250, 10))  # camera feed centered

    # Draw players
    for p in players:
        p.draw(screen)
        p.draw_health_bar(screen)

    pygame.display.flip()
    clock.tick(60)
