#!/usr/bin/env python3
"""
Hand Gesture Mouse Controller & Digital Drawing Tool
=====================================================
A computer vision application that translates hand gestures into mouse controls
and provides a digital drawing interface.

Features:
- Mouse movement via index finger tracking
- Left click (index + thumb pinch)
- Right click (middle + thumb pinch)
- Scroll mode (index + middle fingers up)
- Drawing mode with gesture-based color switching
- Exponential Moving Average (EMA) smoothing
- Multi-threaded video capture for performance

Author: OpenClaw Assistant
"""

import cv2
import mediapipe as mp
import numpy as np
from threading import Thread
from collections import deque
import time
import sys
import os

# Try to import pyautogui with fallback for headless environments
try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except Exception as e:
    print(f"Warning: PyAutoGUI not available ({e})")
    print("Running in visualization-only mode (no mouse control)")
    PYAUTOGUI_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

# Video settings
WEBCAM_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Screen resolution (will auto-detect)
if PYAUTOGUI_AVAILABLE:
    SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
else:
    # Fallback for headless mode
    SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080

# Coordinate mapping with buffer zone (prevents cursor from getting stuck at edges)
BUFFER_ZONE = 0.1  # 10% buffer on each side
MAPPING_X_MIN = int(FRAME_WIDTH * BUFFER_ZONE)
MAPPING_X_MAX = int(FRAME_WIDTH * (1 - BUFFER_ZONE))
MAPPING_Y_MIN = int(FRAME_HEIGHT * BUFFER_ZONE)
MAPPING_Y_MAX = int(FRAME_HEIGHT * (1 - BUFFER_ZONE))

# Gesture thresholds
PINCH_THRESHOLD = 0.05  # Distance threshold for click (normalized)
SCROLL_THRESHOLD = 0.3  # Distance between index and middle fingers for scroll mode
SCROLL_SENSITIVITY = 15  # Scroll speed multiplier

# Smoothing parameters
EMA_ALPHA = 0.3  # Exponential Moving Average factor (0.1-0.5, lower = smoother but more lag)
SMOOTHING_HISTORY = 10  # Number of frames for smoothing

# Drawing settings
DRAWING_COLORS = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'yellow': (0, 255, 255),
    'white': (255, 255, 255)
}
DEFAULT_COLOR = 'blue'
BRUSH_SIZE = 5

# Performance
USE_MULTITHREADING = True

# ============================================================================
# GLOBAL STATE
# ============================================================================

class AppState:
    """Manages application state across threads."""
    def __init__(self):
        self.mode = 'mouse'  # 'mouse', 'drawing', 'scroll'
        self.current_color = DEFAULT_COLOR
        self.drawing_active = False
        self.running = True
        self.frame = None
        self.landmarks = None
        self.handedness = None
        
        # Smoothing buffers
        self.x_history = deque(maxlen=SMOOTHING_HISTORY)
        self.y_history = deque(maxlen=SMOOTHING_HISTORY)
        
        # Drawing canvas (transparent overlay)
        self.canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        
        # Previous positions for draw detection
        self.prev_y = None
        
        # Pinch states for edge detection and double-click
        self.index_thumb_pinched = False
        self.middle_thumb_pinched = False
        self.last_index_pinch_time = 0

state = AppState()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def exponential_moving_average(history, new_value, alpha=EMA_ALPHA):
    """
    Apply Exponential Moving Average smoothing to reduce cursor jitter.
    
    Args:
        history: deque of previous values
        new_value: current measurement
        alpha: smoothing factor (0-1, lower = smoother)
    
    Returns:
        Smoothed value
    """
    if len(history) == 0:
        history.append(new_value)
        return new_value
    
    # EMA formula: smoothed = alpha * new + (1-alpha) * previous
    smoothed = alpha * new_value + (1 - alpha) * history[-1]
    history.append(smoothed)
    return smoothed

def map_coordinates(x, y):
    """
    Map webcam frame coordinates to screen coordinates with buffer zone.
    
    Args:
        x, y: Normalized coordinates (0-1) from MediaPipe
    
    Returns:
        Tuple (screen_x, screen_y)
    """
    # Convert normalized to frame coordinates
    frame_x = int(x * FRAME_WIDTH)
    frame_y = int(y * FRAME_HEIGHT)
    
    # Apply buffer zone mapping
    screen_x = np.interp(frame_x, (MAPPING_X_MIN, MAPPING_X_MAX), (0, SCREEN_WIDTH))
    screen_y = np.interp(frame_y, (MAPPING_Y_MIN, MAPPING_Y_MAX), (0, SCREEN_HEIGHT))
    
    return int(screen_x), int(screen_y)

def calculate_distance(landmark1, landmark2):
    """
    Calculate Euclidean distance between two 3D landmarks.
    
    Args:
        landmark1, landmark2: MediaPipe landmarks with x, y, z
    
    Returns:
        Normalized distance (0-1 range approximately)
    """
    if landmark1 is None or landmark2 is None:
        return float('inf')
    
    return np.sqrt(
        (landmark1.x - landmark2.x) ** 2 +
        (landmark1.y - landmark2.y) ** 2 +
        (landmark1.z - landmark2.z) ** 2
    )

def is_finger_up(landmarks, finger_tip_idx, finger_pip_idx):
    """
    Check if a finger is extended (pointing up).
    
    Args:
        landmarks: Hand landmarks from MediaPipe
        finger_tip_idx: Index of finger tip landmark
        finger_pip_idx: Index of finger PIP joint (second joint from palm)
    
    Returns:
        True if finger is up
    """
    if landmarks is None:
        return False
    
    tip = landmarks[finger_tip_idx]
    pip = landmarks[finger_pip_idx]
    
    # Finger is up if tip is above PIP joint (smaller y value in image coordinates)
    return tip.y < pip.y

# ============================================================================
# GESTURE RECOGNITION
# ============================================================================

def detect_gestures(landmarks, handedness):
    """
    Detect hand gestures and update application state.
    
    Args:
        landmarks: Hand landmarks from MediaPipe
        handedness: Which hand (Left/Right)
    
    Returns:
        Dictionary with gesture states
    """
    if landmarks is None:
        return {'mode': state.mode, 'action': None}
    
    gestures = {
        'mode': state.mode,
        'action': None,
        'color_change': None
    }
    
    # Landmark indices (MediaPipe Hand Landmarks)
    THUMB_TIP = 4
    INDEX_TIP = 8
    INDEX_PIP = 6
    MIDDLE_TIP = 12
    MIDDLE_PIP = 10
    RING_TIP = 16
    RING_PIP = 14
    PINKY_TIP = 20
    PINKY_PIP = 18
    
    # Check finger states
    index_up = is_finger_up(landmarks, INDEX_TIP, INDEX_PIP)
    middle_up = is_finger_up(landmarks, MIDDLE_TIP, MIDDLE_PIP)
    ring_up = is_finger_up(landmarks, RING_TIP, RING_PIP)
    pinky_up = is_finger_up(landmarks, PINKY_TIP, PINKY_PIP)
    
    # Calculate distances
    index_thumb_dist = calculate_distance(landmarks[INDEX_TIP], landmarks[THUMB_TIP])
    middle_thumb_dist = calculate_distance(landmarks[MIDDLE_TIP], landmarks[THUMB_TIP])
    
    is_index_thumb_pinched = index_thumb_dist < PINCH_THRESHOLD
    is_middle_thumb_pinched = middle_thumb_dist < PINCH_THRESHOLD
    
    # Mode switching: Toggle drawing mode with all fingers up (open palm)
    if index_up and middle_up and ring_up and pinky_up:
        # Hold for a moment to toggle (prevent accidental switches)
        if state.prev_y is None:
            state.prev_y = time.time()
        elif time.time() - state.prev_y > 1.5:
            if state.mode == 'mouse':
                state.mode = 'drawing'
                state.drawing_active = True
            elif state.mode == 'drawing':
                state.mode = 'mouse'
                state.drawing_active = False
            state.prev_y = None
    else:
        state.prev_y = None
    
    # Drawing mode logic
    if state.mode == 'drawing':
        # Eraser mode: Index and thumb pinched together
        if is_index_thumb_pinched:
            gestures['action'] = 'erase'
            
        # Change color: Thumb and middle finger pinched once (edge detection)
        elif is_middle_thumb_pinched and not state.middle_thumb_pinched:
            color_order = ['blue', 'red', 'green', 'yellow', 'white']
            current_idx = color_order.index(state.current_color) if state.current_color in color_order else 0
            next_color = color_order[(current_idx + 1) % len(color_order)]
            gestures['color_change'] = next_color
            gestures['action'] = 'color_change'
            
        # Drawing with index finger (and no pinch)
        elif index_up and not middle_up and not ring_up and not pinky_up and not is_index_thumb_pinched:
            gestures['action'] = 'draw'
            
    # Mouse mode (default)
    else:
        state.mode = 'mouse'
        
        # Right click / options: thumb and middle pinch (edge detection)
        if is_middle_thumb_pinched and not state.middle_thumb_pinched:
            gestures['action'] = 'right_click'
            
        # Left click or Double click: thumb and index pinch (edge detection)
        elif is_index_thumb_pinched and not state.index_thumb_pinched:
            current_time = time.time()
            if current_time - state.last_index_pinch_time < 0.5:
                gestures['action'] = 'double_click'
                state.last_index_pinch_time = 0  # reset to prevent triple click
            else:
                gestures['action'] = 'left_click'
                state.last_index_pinch_time = current_time
                
        # Move mouse: only index finger is up
        elif index_up and not middle_up and not ring_up and not pinky_up and not is_index_thumb_pinched:
            gestures['action'] = 'move'
            
    # Update pinch states for next frame
    state.index_thumb_pinched = is_index_thumb_pinched
    state.middle_thumb_pinched = is_middle_thumb_pinched
    
    return gestures

# ============================================================================
# MOUSE CONTROL
# ============================================================================

def control_mouse(landmarks, gestures):
    """
    Control system mouse based on hand landmarks and gestures.
    
    Args:
        landmarks: Hand landmarks
        gestures: Detected gestures dictionary
    """
    if not PYAUTOGUI_AVAILABLE:
        return  # Skip mouse control in headless mode
    
    if landmarks is None:
        return
    
    INDEX_TIP = 8
    
    # Get index finger tip position
    x = landmarks[INDEX_TIP].x
    y = landmarks[INDEX_TIP].y
    
    # Apply smoothing
    smoothed_x = exponential_moving_average(state.x_history, x)
    smoothed_y = exponential_moving_average(state.y_history, y)
    
    # Map to screen coordinates
    screen_x, screen_y = map_coordinates(smoothed_x, smoothed_y)
    
    # Execute actions
    if gestures.get('action') in ['move', 'left_click', 'double_click', 'right_click']:
        # Move mouse only if action is one of the valid movement/click actions
        pyautogui.moveTo(screen_x, screen_y, duration=0.1)
        
        if gestures['action'] == 'left_click':
            pyautogui.click(button='left')
        elif gestures['action'] == 'double_click':
            pyautogui.doubleClick()
        elif gestures['action'] == 'right_click':
            pyautogui.click(button='right')

# ============================================================================
# DRAWING LOGIC
# ============================================================================

def draw_on_canvas(landmarks, action='draw'):
    """
    Draw on the virtual canvas using index finger position.
    
    Args:
        landmarks: Hand landmarks
        action: 'draw' or 'erase'
    """
    if landmarks is None:
        return
    
    INDEX_TIP = 8
    
    # Get index finger tip position in frame coordinates
    x = int(landmarks[INDEX_TIP].x * FRAME_WIDTH)
    y = int(landmarks[INDEX_TIP].y * FRAME_HEIGHT)
    
    if action == 'erase':
        # Eraser: larger circle, removes color (draws background color)
        # We'll draw with a large black circle to simulate erasing
        # Actually, we need to clear the canvas area
        eraser_size = 20
        cv2.circle(state.canvas, (x, y), eraser_size, (0, 0, 0), -1)
    else:
        # Drawing mode
        # Get current color
        color = DRAWING_COLORS.get(state.current_color, DRAWING_COLORS[DEFAULT_COLOR])
        
        # Draw on canvas
        cv2.circle(state.canvas, (x, y), BRUSH_SIZE, color, -1)

# ============================================================================
# VISUALIZATION
# ============================================================================

def draw_overlay(frame, landmarks, handedness, gestures):
    """
    Draw landmarks, mode indicator, and feedback on the video frame.
    
    Args:
        frame: Video frame from webcam
        landmarks: Hand landmarks
        handedness: Left/Right hand
        gestures: Current gestures
    
    Returns:
        Frame with overlays
    """
    output = frame.copy()
    
    # Draw canvas overlay in drawing mode
    if state.mode == 'drawing':
        # Blend canvas with frame
        alpha = 0.5
        cv2.addWeighted(state.canvas, alpha, output, 1 - alpha, 0, output)
    
    # Draw landmarks if detected
    if landmarks is not None:
        # Get hand connections - API changed in newer MediaPipe versions
        hand_connections = mp.solutions.hands.HAND_CONNECTIONS
        
        # HAND_CONNECTIONS is now a frozenset directly in newer versions
        connections_list = list(hand_connections) if isinstance(hand_connections, (frozenset, set)) else hand_connections.connections
        
        # Draw connections
        for connection in connections_list:
            point1 = landmarks[connection[0]]
            point2 = landmarks[connection[1]]
            
            pt1 = (int(point1.x * FRAME_WIDTH), int(point1.y * FRAME_HEIGHT))
            pt2 = (int(point2.x * FRAME_WIDTH), int(point2.y * FRAME_HEIGHT))
            
            cv2.line(output, pt1, pt2, (0, 255, 0), 2)
        
        # Draw landmarks
        for idx, landmark in enumerate(landmarks):
            x = int(landmark.x * FRAME_WIDTH)
            y = int(landmark.y * FRAME_HEIGHT)
            
            # Color-code key landmarks
            if idx == 8:  # Index tip
                color = (0, 0, 255)  # Red
            elif idx == 4:  # Thumb tip
                color = (255, 0, 0)  # Blue
            elif idx == 12:  # Middle tip
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 255, 0)  # Green
            
            cv2.circle(output, (x, y), 5, color, -1)
    
    # Draw mode indicator
    mode_colors = {
        'mouse': (0, 255, 0),
        'drawing': (255, 0, 0),
        'scroll': (0, 255, 255)
    }
    
    mode_color = mode_colors.get(state.mode, (255, 255, 255))
    mode_text = f"Mode: {state.mode.upper()}"
    
    if state.mode == 'drawing':
        mode_text += f" | Color: {state.current_color.upper()}"
    
    cv2.putText(output, mode_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
    
    # Draw action indicator
    if gestures['action']:
        action_text = f"Action: {gestures['action'].replace('_', ' ').title()}"
        cv2.putText(output, action_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw instructions
    instructions = [
        "Q: Quit | P: Pause | C: Clear Canvas",
        "Open palm (1.5s): Toggle Draw Mode",
        "Draw: Index=Draw | Pinch=Erase | Mid+Thumb=Color"
    ] if state.mode == 'drawing' else [
        "Q: Quit | P: Pause | C: Clear Canvas",
        "Open palm (1.5s): Toggle Draw Mode",
        "Mouse: Index=Move | Pinch=Sel/Opn | Mid+Thumb=Opts"
    ]
    
    for i, instr in enumerate(instructions):
        cv2.putText(output, instr, (10, FRAME_HEIGHT - 50 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Draw FPS
    fps_text = f"FPS: {int(1000 / max(1, state.fps_delta))}" if hasattr(state, 'fps_delta') else "FPS: --"
    cv2.putText(output, fps_text, (FRAME_WIDTH - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return output

# ============================================================================
# VIDEO CAPTURE (Multi-threaded)
# ============================================================================

class VideoCaptureThread(Thread):
    """Multi-threaded video capture for better performance."""
    
    def __init__(self, webcam_id=WEBCAM_ID, width=FRAME_WIDTH, height=FRAME_HEIGHT):
        super().__init__(daemon=True)
        self.webcam_id = webcam_id
        self.width = width
        self.height = height
        self.frame = None
        self.running = True
    
    def run(self):
        """Capture video frames in a separate thread."""
        cap = cv2.VideoCapture(self.webcam_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        
        while self.running and state.running:
            ret, frame = cap.read()
            if ret:
                self.frame = frame
            else:
                print("Warning: Failed to capture frame")
        
        cap.release()
    
    def get_frame(self):
        """Get the latest captured frame."""
        return self.frame
    
    def stop(self):
        """Stop the capture thread."""
        self.running = False

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application loop."""
    print("=" * 60)
    print("Hand Gesture Mouse Controller & Drawing Tool")
    print("=" * 60)
    print(f"Screen Resolution: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
    print(f"Webcam Resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"Buffer Zone: {BUFFER_ZONE * 100:.0f}%")
    print("\nControls:")
    print("  - Index finger: Move cursor")
    print("  - Closed hand: No action (cursor freezes)")
    print("  - Index + Thumb pinch once: Left click / Select")
    print("  - Index + Thumb pinch twice: Double click / Open")
    print("  - Middle + Thumb pinch: Right click / Options")
    print("  - Open palm (1.5s): Toggle drawing mode")
    print("  - Drawing mode: Index finger draws")
    print("  - Drawing mode: Index + Thumb pinch to erase")
    print("  - Drawing mode: Middle + Thumb pinch to change color")
    print("\nPress 'q' to quit, 'p' to pause\n")
    print("=" * 60)
    
    # Initialize video capture
    if USE_MULTITHREADING:
        print("Starting multi-threaded video capture...")
        video_thread = VideoCaptureThread()
        video_thread.start()
        time.sleep(0.5)  # Wait for first frame
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    # Initialize drawing
    mp_drawing = mp.solutions.drawing_utils
    
    last_time = time.time()
    paused = False
    
    try:
        while state.running:
            # Calculate FPS
            current_time = time.time()
            state.fps_delta = (current_time - last_time) * 1000
            last_time = current_time
            
            # Get frame
            if USE_MULTITHREADING:
                frame = video_thread.get_frame()
            else:
                # Fallback to single-threaded capture
                cap = cv2.VideoCapture(WEBCAM_ID)
                ret, frame = cap.read()
                cap.release()
            
            if frame is None:
                print("Error: No frame captured")
                continue
            
            # Resize frame
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            frame = cv2.flip(frame, 1)  # Mirror for natural interaction
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if not paused:
                # Process hand detection
                results = hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    # Get first detected hand
                    state.landmarks = results.multi_hand_landmarks[0].landmark
                    # Fix: API changed in newer MediaPipe versions
                    if results.multi_handedness:
                        state.handedness = results.multi_handedness[0].classification[0].label
                    else:
                        state.handedness = "Unknown"
                    
                    # Detect gestures
                    gestures = detect_gestures(state.landmarks, state.handedness)
                    
                    # Update color if changed
                    if gestures.get('color_change'):
                        state.current_color = gestures['color_change']
                    
                    # Control mouse or draw
                    if state.mode == 'mouse' or state.mode == 'scroll':
                        control_mouse(state.landmarks, gestures)
                    elif state.mode == 'drawing':
                        action = gestures.get('action')
                        if action in ['draw', 'erase']:
                            draw_on_canvas(state.landmarks, action)
                    
                    # Draw overlay
                    frame = draw_overlay(frame, state.landmarks, state.handedness, gestures)
                else:
                    state.landmarks = None
                    state.handedness = None
                    frame = draw_overlay(frame, None, None, {'action': None})
            else:
                # Paused - just show frame with pause indicator
                cv2.putText(frame, "PAUSED", (FRAME_WIDTH // 2 - 80, FRAME_HEIGHT // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            # Show frame
            cv2.imshow('Hand Gesture Controller', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nShutting down...")
                break
            elif key == ord('p'):
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")
            elif key == ord('c'):
                # Clear canvas
                state.canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                print("Canvas cleared")
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        state.running = False
        if USE_MULTITHREADING:
            video_thread.stop()
            video_thread.join()
        
        hands.close()
        cv2.destroyAllWindows()
        
        print("\nApplication closed successfully")

# ============================================================================
# FAIL-SAFE
# ============================================================================

def fail_safe():
    """
    Emergency fail-safe: Move mouse to corner and exit.
    Can be triggered by moving mouse to top-left corner rapidly.
    """
    if not PYAUTOGUI_AVAILABLE:
        print("Fail-safe: PyAutoGUI not available, exiting...")
        sys.exit(0)
    
    pyautogui.moveTo(0, 0)
    pyautogui.moveTo(SCREEN_WIDTH, SCREEN_HEIGHT)
    pyautogui.moveTo(0, 0)
    print("Fail-safe triggered!")
    sys.exit(0)

# Check for fail-safe on startup (optional)
# Uncomment the line below to enable automatic fail-safe detection
# fail_safe()

if __name__ == "__main__":
    main()
