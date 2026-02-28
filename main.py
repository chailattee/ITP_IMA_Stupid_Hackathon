import pygame
import sys
import cv2
from threading import Thread
from queue import Queue
from imutils.video import VideoStream
import imutils
from face_utils import get_mouth_loc_with_height, draw_mouth

# Frame queue for thread-safe communication
frame_queue = Queue(maxsize=2)
running = True
last_frame = None  # Store last frame to prevent flashing

def camera_capture_thread():
    """Captures frames from camera with AI processing"""
    global running
    vs = VideoStream(0).start()
    
    while running:
        frame = vs.read()
        enhanced = cv2.detailEnhance(frame, sigma_s=10, sigma_r=0.15)
        frame = imutils.resize(frame, width=300) #sizing camera feed
        
        # Try AI processing
        result = get_mouth_loc_with_height(enhanced)
        if "error" not in result:
            shape = result['shape']
            frame = draw_mouth(frame, shape)
        
        # Convert BGR to RGB for pygame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = frame_rgb.swapaxes(0, 1)  # Swap to (width, height, channels)
        
        if not frame_queue.full():
            frame_queue.put(frame_rgb)
    
    vs.release()

pygame.init()
gridWidth = 800
gridHeight = 600

screen = pygame.display.set_mode((gridWidth, gridHeight))
clock = pygame.time.Clock()

# Start camera thread
cam_thread = Thread(target=camera_capture_thread, daemon=True)
cam_thread.start()

class player:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def move(self, dx):
        self.x += dx

    def attack(self):
        print("Attack!")
    
    def draw(self):
        #IMPLEMNT LATER
        pass

player1 = player(100, 500)
player2 = player(700, 500)
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            pygame.quit()
            sys.exit()

    screen.fill((255, 255, 255))
    
    # Display camera frame with AI
    if not frame_queue.empty():
        last_frame = frame_queue.get()

    if last_frame is not None:
        surf = pygame.surfarray.make_surface(last_frame)
        screen.blit(surf, (250, 10))  # camera feed 
    pygame.draw.rect(screen, (100, 255, 0), (player1.x, player1.y, 50, 50))  # Player 1
    pygame.draw.rect(screen, (0, 255, 100), (player2.x, player2.y, 50, 50))  # Player 2
    
    pygame.display.flip()
    clock.tick(60)