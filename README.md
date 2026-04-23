# Hand Gesture Mouse Controller & Digital Drawing Tool

A Python-based computer vision application that replaces your standard mouse with hand gestures using your webcam. Control your cursor, click, drag, and even draw digitally on the screen just by moving your hand!

## Features

- **Mouse Movement**: Move your cursor by pointing your index finger.
- **Clicking Mechanics**:
  - **Single Left Click**: Quick pinch of your index and thumb.
  - **Double Left Click**: Double quick pinch of your index and thumb.
  - **Right Click**: Pinch your middle finger and thumb.
- **Drag Mode**: Pinch your index and thumb and hold for 1.5 seconds. The cursor will freeze temporarily while holding, then you can drag items across the screen.
- **Drawing Mode**:
  - Toggle drawing mode by keeping an open palm for 1.5 seconds.
  - Use your index finger to draw on a virtual canvas.
  - Change brush colors by pinching your middle finger and thumb.
  - Erase by pinching your index finger and thumb.

## Installation

It is recommended to run this project in an isolated Python Virtual Environment (`venv`) to prevent dependency conflicts.

### 1. Clone the repository

```bash
git clone https://github.com/NaumanAhmad2005/gesture-controller.git
cd gesture-controller
```

### 2. Create and Activate a Virtual Environment

**On Linux/macOS:**
```bash
python3 -m venv myvenv
source myvenv/bin/activate
```

**On Windows:**
```bash
python -m venv myvenv
myvenv\Scripts\activate
```

### 3. Install Requirements

With your virtual environment activated, install the necessary dependencies:

```bash
pip install -r requirements.txt
```

### 4. Run the Controller

```bash
python gesture_mouse_controller.py
```

*Note: Press `q` to quit the application and `p` to pause.*

## Credits

Created by **naumanf25**.
