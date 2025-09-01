# Configuration file for Multi-Modal Tracking Application

# Camera settings
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_INDEX = 0  # Change this if you have multiple cameras

# Tracking confidence thresholds
BODY_DETECTION_CONFIDENCE = 0.5
BODY_TRACKING_CONFIDENCE = 0.5

HAND_DETECTION_CONFIDENCE = 0.7
HAND_TRACKING_CONFIDENCE = 0.5
MAX_HANDS = 2

FACE_DETECTION_CONFIDENCE = 0.5
FACE_TRACKING_CONFIDENCE = 0.5

# Display settings
SHOW_FPS = True
SHOW_MODE_STATUS = True
SHOW_TRACKING_INFO = True

# Colors (BGR format)
COLORS = {
    'body': (0, 255, 0),      # Green
    'hands': (255, 0, 0),     # Blue  
    'eyes': (0, 0, 255),      # Red
    'head': (255, 255, 0),    # Cyan
    'inactive': (128, 128, 128)  # Gray
}

# Tracking modes (initial state)
DEFAULT_MODES = {
    'body': True,
    'hands': True,
    'eyes': True,
    'head': True
}

# Performance settings
PROCESS_EVERY_N_FRAMES = 1  # Process every frame by default
ENABLE_SMOOTHING = True
SMOOTHING_FACTOR = 0.8

# File paths
SAVE_TRACKING_DATA = False
TRACKING_DATA_PATH = "tracking_data.json"
SAVE_VIDEO = False
VIDEO_OUTPUT_PATH = "tracking_output.mp4"

# Audio feedback settings
ENABLE_AUDIO_FEEDBACK = False
AUDIO_FEEDBACK_VOLUME = 0.5
