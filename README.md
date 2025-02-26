# BAYMAX-GUI: Medical Robot Arm Control System

A computer vision-powered robotic arm control system for medical applications. This system uses deep learning to segment and identify acute traumatic injuries and provides precise robotic arm positioning to target specific points of interest.

## Overview

BAYMAX-GUI integrates computer vision and robotics to create an automated solution for identifying and targeting injury sites. The system consists of:

1. A vision module using a custom-trained segmentation model to detect 
and outline wound boundaries
2. An inverse kinematics solver that calculates optimal joint positions for a 4-DOF robotic arm
3. A control interface that visualizes both the camera feed and robotic arm positioning
4. Arduino communication to control physical hardware

## Key Features

- **Real-time injury segmentation** using a custom-trained TensorFlow/Keras U-Net model
- **Automated target point extraction** identifying left, right, top, and bottom coordinates of detected wounds
- **Advanced inverse kinematics solver** using Jacobian matrices with constraints for optimal arm positioning
- **Dual-plane visualization** showing both side (X-Z) and top (X-Y) views of the robotic arm position
- **Intuitive GUI** built with PyQt5 providing camera feed, arm visualization, and control options
- **Edge device compatibility** designed to run on Raspberry Pi with Arduino hardware control
- **Complete system independence** functions as a standalone unit without requiring external processing

## Technical Details

### Computer Vision Module

The system uses a custom-trained segmentation model to identify wound boundaries in real-time:

- Processes webcam input at 30 FPS
- Applies U-Net architecture with custom loss functions (Dice coefficient and IoU)
- Implements morphological operations to clean and refine segmentation masks
- Extracts extreme points (left, right, top, bottom) for targeting

### Kinematics Engine

The robotic arm positioning is calculated using:

- Forward kinematics to model the arm's position in 3D space
- Jacobian-based inverse kinematics to solve for joint angles
- Dynamic angle constraints based on target distance
- Iterative solution finding with convergence tolerance

```python
def jacobian_inverse_kinematics(target_x, target_z, theta1, theta2, theta3, L1, L2, L3, max_iter=500, tol=0.001):
    # Initial angles
    angles = np.array([theta1, theta2, theta3])
    
    # Define constraints based on target distance
    planar_dist = abs(target_x)
    
    # Iteratively solve for angles that reach the target
    for i in range(max_iter):
        # ...
        if np.linalg.norm(error) < tol:
            break
        # ...
        
    return True, servo_theta1, servo_theta2, servo_theta3
```

### Hardware Communication

The system sends calculated joint positions to an Arduino:

- Converts radian joint angles to servo-compatible degree values
- Formats and transmits serial commands with timeout handling
- Processes acknowledgment from Arduino controller
- Manages connection state with automatic reset and error handling

## Hardware Requirements

- Raspberry Pi 5 (or equivalent computer)
- Arduino Uno Rev3
- USB webcam or Raspberry Pi camera
- 4-DOF robotic arm with servos
- USB cable for Arduino connection

## Software Requirements

- Python 3.7+
- TensorFlow 2.x
- PyQt5
- OpenCV
- NumPy
- Matplotlib
- PySerial

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/BAYMAX-GUI.git
   cd BAYMAX-GUI
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Connect hardware:
   - Attach webcam to Raspberry Pi
   - Connect Arduino via USB
   - Connect servo motors to appropriate Arduino pins

4. Upload Arduino sketch:
   ```bash
   arduino-cli upload -p /dev/ttyACM0 -b arduino:avr:uno arduino/servo_controller
   ```

5. Run the application:
   ```bash
   python main.py
   ```

## Usage

1. **Wound Detection**: The system automatically detects wound boundaries when they appear in the camera feed
2. **Target Selection**: Select a target point (Left, Right, Top, Bottom) using the dropdown menu or number keys (1-4)
3. **Arduino Connection**: Toggle the Arduino connection with the "Toggle Arduino" button or by pressing 'a'
4. **Arm Positioning**: The system will calculate and display optimal arm positions for the selected target
5. **Hardware Control**: When connected, joint positions will be sent to the Arduino to move the physical arm

## Key Files

- `main.py`: Main application entry point
- `process_frame()`: Computer vision and segmentation functions
- `jacobian_inverse_kinematics()`: Core inverse kinematics solver
- `forward_kinematics()`: Forward kinematics calculations
- `send_to_arduino()`: Hardware communication functions
- `MainWindow`: PyQt5 interface implementation
- `OpenCVApp`: Core application logic

## Future Development

- Enhanced segmentation models for varied wound types
- Path planning for obstacle avoidance
- Expanded tactile feedback system
- Multiple wound tracking and prioritization
- Teleoperation capabilities

## License

[Your license information here]

## Acknowledgments

- This project utilizes custom implementations of segmentation models, inverse kinematics, and hardware control systems
- Development supported by [your affiliations/acknowledgments]

## DEMO SAMPLES BELOW
## NSFW WARNING: TEST SAMPLES USED HERE ARE IMAGES OF REAL OPEN WOUNDS CONTAINING BLOOD

<img width="913" alt="Screenshot 2025-02-26 at 5 20 23 AM" src="https://github.com/user-attachments/assets/abdc0b96-5b53-4b73-9a18-efd4d563cbcb" />
<img width="911" alt="Screenshot 2025-02-26 at 5 21 29 AM" src="https://github.com/user-attachments/assets/bfa4726f-7d0a-444c-98a4-aeb67a8c394f" />
<img width="916" alt="Screenshot 2025-02-26 at 5 23 03 AM" src="https://github.com/user-attachments/assets/8e8103f4-3538-4ba5-b7e1-d7819fb7c61e" />
