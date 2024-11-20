import pygame
import time

# Initialize pygame modules
pygame.init()  # Initializes all pygame modules
pygame.mixer.init()  # Initialize the mixer module separately

# Check if the pygame mixer is initialized properly
if pygame.mixer.get_init() is None:
    print("Pygame mixer failed to initialize.")
else:
    print("Pygame mixer initialized successfully.")

# Load and play a sound (for example, 'output.mp3')
pygame.mixer.music.load("output.mp3")
pygame.mixer.music.play()

# Wait until the sound finishes playing
while pygame.mixer.music.get_busy():
    time.sleep(1)
