import sys
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.utils import register_keras_serializable
from keras import backend as K
import matplotlib
matplotlib.use('Agg')  # Use Agg backend (no GUI)
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import math
import time
import serial
import threading
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QStatusBar, QComboBox)


# Model and weights
background_ratio = 0.8
wound_ratio = 0.2
class_weights = {0: 1.0, 1: background_ratio / wound_ratio}

@register_keras_serializable()
def dice_coef(y_true, y_pred, smooth=10e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice

@register_keras_serializable()
def weighted_dice_coef(y_true, y_pred, smooth=10e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    weights = K.gather(K.constant(list(class_weights.values())), K.cast(y_true_f, dtype='int32'))
    intersection = K.sum(y_true_f * y_pred_f * weights)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f * weights) + K.sum(y_pred_f * weights) + smooth)
    return dice

@register_keras_serializable()
def dice_coef_loss(y_true, y_pred, smooth=10e-6):
    return 1 - dice_coef(y_true, y_pred, smooth)

@register_keras_serializable()
def weighted_dice_coef_loss(y_true, y_pred, smooth=10e-6):
    dice_coef = weighted_dice_coef(y_true, y_pred, smooth)
    loss = 1 - dice_coef
    return loss

@register_keras_serializable()
def iou(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou = (intersection + 1.0) / (union + 1.0)
    return iou

@register_keras_serializable()
def iou_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return 1 - iou

@register_keras_serializable()
def combined_dice_iou_loss(y_true, y_pred, smooth=1e-6, dice_weight=0.5, iou_weight=0.5, wound_weight=0.75):
    unweighted_dice_loss = dice_coef_loss(y_true, y_pred, smooth)
    weighted_dice_loss = weighted_dice_coef_loss(y_true, y_pred, smooth)
    combined_dice_loss = wound_weight * weighted_dice_loss + (1 - wound_weight) * unweighted_dice_loss
    iou_loss_val = iou_loss(y_true, y_pred, smooth)
    combined_loss = dice_weight * combined_dice_loss + iou_weight * iou_loss_val
    return combined_loss

def process_frame(frame, model):
    # Store original dimensions
    original_height, original_width = frame.shape[:2]
    
    # Preprocess the frame for model
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(processed_frame, (256, 256))
    model_input = img_to_array(resized_frame) / 255.0
    model_input = np.expand_dims(model_input, axis=0)
    
    # Make predictions
    prediction = model.predict(model_input, verbose=0)
    binary_mask = (prediction[0, :, :, 0] > 0.99).astype(np.uint8)
    
    # Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    # Find and filter contours on the 256x256 mask
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 50
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    # Scale contours back to original image size
    scale_x = original_width / 256
    scale_y = original_height / 256
    
    scaled_contours = []
    for contour in filtered_contours:
        scaled_contour = contour.copy()
        scaled_contour[:, :, 0] = contour[:, :, 0] * scale_x
        scaled_contour[:, :, 1] = contour[:, :, 1] * scale_y
        scaled_contours.append(scaled_contour)
    
    return scaled_contours, frame

def forward_kinematics(base_rotation, theta1, theta2, theta3, L1, L2, L3, base_height):
    # Convert servo angles to mechanical angles - keeping original orientations
    theta2 = -theta2 + np.pi/2  # This matches the physical servo mounting
    theta3 = np.pi/2 - theta3   # This matches the physical servo mounting
    
    # Calculate positions using mechanical angles
    x1 = L1 * np.cos(theta1)
    z1 = L1 * np.sin(theta1) + base_height
    
    x2 = x1 + L2 * np.cos(theta1 + theta2)
    z2 = z1 + L2 * np.sin(theta1 + theta2)
    
    x3 = x2 + L3 * np.cos(theta1 + theta2 + theta3)
    z3 = z2 + L3 * np.sin(theta1 + theta2 + theta3)
    
    # Apply base rotation
    rotated_x = [0, x1 * np.cos(base_rotation), x2 * np.cos(base_rotation), x3 * np.cos(base_rotation)]
    rotated_y = [0, x1 * np.sin(base_rotation), x2 * np.sin(base_rotation), x3 * np.sin(base_rotation)]
    z = [base_height, z1, z2, z3]
    
    return rotated_x, rotated_y, z

def compute_jacobian(theta1, theta2, theta3, L1, L2, L3):
    # Compute each element of the Jacobian matrix
    J = np.zeros((2, 3))  # 2x3 Jacobian for x,z coordinates
    
    # For x coordinate
    J[0,0] = -L1*np.sin(theta1) - L2*np.sin(theta1 + theta2) - L3*np.sin(theta1 + theta2 + theta3)
    J[0,1] = -L2*np.sin(theta1 + theta2) - L3*np.sin(theta1 + theta2 + theta3)
    J[0,2] = -L3*np.sin(theta1 + theta2 + theta3)
    
    # For z coordinate
    J[1,0] = L1*np.cos(theta1) + L2*np.cos(theta1 + theta2) + L3*np.cos(theta1 + theta2 + theta3)
    J[1,1] = L2*np.cos(theta1 + theta2) + L3*np.cos(theta1 + theta2 + theta3)
    J[1,2] = L3*np.cos(theta1 + theta2 + theta3)
    
    return J

def jacobian_inverse_kinematics(target_x, target_z, theta1, theta2, theta3, L1, L2, L3, max_iter=500, tol=0.001):
    
    angles = np.array([theta1, theta2, theta3])
    
    # Get planar distance to target
    planar_dist = abs(target_x)  # Since we're in a 2D plane now
    
    # Define theta1 constraints based on planar distance
    if planar_dist <= 110:
        theta1_min, theta1_max = np.radians(110), np.radians(180)
        theta2_min, theta2_max = np.radians(90), np.radians(270)
        theta3_min, theta3_max = np.radians(0), np.radians(360)  # More vertical for close targets
    elif planar_dist <= 150:
        theta1_min, theta1_max = np.radians(75), np.radians(110)
        theta2_min, theta2_max = np.radians(90), np.radians(270)
        theta3_min, theta3_max = np.radians(0), np.radians(270)   # Mid-range
    elif planar_dist <= 250:
        theta1_min, theta1_max = np.radians(30), np.radians(75)
        theta2_min, theta2_max = np.radians(90), np.radians(270)
        theta3_min, theta3_max = np.radians(0), np.radians(270)
    elif planar_dist <= 300:
        theta1_min, theta1_max = np.radians(10), np.radians(60)
        theta2_min, theta2_max = np.radians(90), np.radians(360)
        theta3_min, theta3_max = np.radians(90), np.radians(90)     # Longer reach
    else:
        theta1_min, theta1_max = np.radians(0), np.radians(90)
        theta2_min, theta2_max = np.radians(0), np.radians(270)
        theta3_min, theta3_max = np.radians(0), np.radians(270)    # Maximum reach
    
    print(f"Planar distance: {planar_dist:.2f}")
    print(f"Theta1 constraints: {np.degrees(theta1_min):.2f}° to {np.degrees(theta1_max):.2f}°")
    print(f"Theta2 constraints: {np.degrees(theta2_min):.2f}° to {np.degrees(theta2_max):.2f}°")
    print(f"Theta3 constraints: {np.degrees(theta3_min):.2f}° to {np.degrees(theta3_max):.2f}°")
    
    for i in range(max_iter):
        theta1, theta2, theta3 = angles
        
        # Current end effector position
        x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2) + L3 * np.cos(theta1 + theta2 + theta3)
        z = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2) + L3 * np.sin(theta1 + theta2 + theta3)
        
        # Error vector
        error = np.array([[target_x - x], [target_z - z]])  # 2x1 column vector
        if np.linalg.norm(error) < tol:
            break
            
        # Compute Jacobian
        J = compute_jacobian(theta1, theta2, theta3, L1, L2, L3)
        
        # Compute pseudo-inverse using Moore-Penrose
        damped_pinv = np.linalg.pinv(J) 
        
        # Update angles
        delta_theta = np.dot(damped_pinv, error).flatten()  # Ensure 1D array
        angles += delta_theta * 0.5  # Apply damping factor
        
        # Apply angle constraints dynamically
        angles[0] = np.clip(angles[0], theta1_min, theta1_max)  # Theta1
        angles[1] = np.clip(angles[1], theta2_min, theta2_max)  # Theta2
        #angles[2] = np.clip(angles[2], theta3_min, theta3_max)  # Theta3
        
    # Convert to servo angles
    theta1, theta2, theta3 = angles
    
    # Convert mechanical angles to servo angles
    servo_theta1 = theta1  # theta1 doesn't change
    servo_theta2 = np.pi/2 - theta2  # Reverse of -theta2 + np.pi/2
    servo_theta3 = np.pi/2 - theta3  # This is already correct in your current code

    print(f"Theta 1: {np.degrees(theta1)}")
    print(f"Theta 2: {np.degrees(theta2)}")
    print(f"Theta 3: {np.degrees(theta3)}")

    print(f"Servo Theta 1: {np.degrees(servo_theta1)}")
    print(f"Servo Theta 2: {np.degrees(servo_theta2)}, +{np.degrees(servo_theta2)+360}")
    print(f"Servo Theta 3: {np.degrees(servo_theta3)}, +{np.degrees(servo_theta3)+360}")

    print(f"Shoulder: {np.degrees(theta1)}")
    print(f"Elbow: {np.degrees(servo_theta2)+360}")
    print(f"wrist: {np.degrees(servo_theta3)+360}")
    
    # Verify the solution
    final_x, final_y, final_z = forward_kinematics(0, theta1, theta2, theta3, L1, L2, L3, 0)
    print(f"Target: ({target_x}, {target_z})")
    print(f"Achieved: ({final_x[-1]}, {final_z[-1]})")
    print(f"Error: {np.sqrt((final_x[-1] - target_x)**2 + (final_z[-1] - target_z)**2)}")
    print(f"Iterations: {i+1}")

    return True, servo_theta1, servo_theta2, servo_theta3

def new_inverse_kinematics_3dof(x, y, z, L1, L2, L3, base_height, x_base):
    try:
        # Handle base rotation separately
        base_rotation = np.arctan2(y, x)
        if base_rotation < 0:
            base_rotation += 2*np.pi
            
        # Get planar distance
        planar_dist = np.sqrt(x**2 + y**2)
        z_adjusted = z - base_height
        
        # Set initial angles based on planar distance
        if planar_dist <= 110:
            init_theta1 = np.radians(160)  # More vertical initial position
            init_theta2 = np.radians(215)   # Start with elbow up
            init_theta3 = np.radians(210)    # Initial end effector angle
        elif planar_dist <= 150:
            init_theta1 = np.radians(160)  # More horizontal start
            init_theta2 = np.radians(215)
            init_theta3 = np.radians(210)
        elif planar_dist <= 200:
            init_theta1 = np.radians(160)  # More horizontal start
            init_theta2 = np.radians(215)
            init_theta3 = np.radians(210)
        else:
            init_theta1 = np.radians(160)  # More horizontal start
            init_theta2 = np.radians(215)
            init_theta3 = np.radians(210)
        
        # Solve 2D problem in the plane
        success, servo_theta1, servo_theta2, servo_theta3 = jacobian_inverse_kinematics(
            planar_dist, z_adjusted, 
            init_theta1, init_theta2, init_theta3,
            L1, L2, L3
        )
        
        if success:
            return True, base_rotation, servo_theta1, servo_theta2, servo_theta3
        return False, 0, 0, 0, 0
        
    except Exception as e:
        print(f"Inverse kinematics error: {e}")
        return False, 0, 0, 0, 0

def plot_arm(ax_side, ax_top, x, y, z, base_height, base_rotation):
    ax_side.clear()
    ax_top.clear()
    
    # Side view
    ax_side.plot([0, 0], [0, base_height], 'k-', lw=2)
    ax_side.plot(x, z, 'o-', lw=2)
    
    x_margin = max(abs(min(x)), abs(max(x))) * 1.2
    ax_side.set_xlim(-x_margin, x_margin)
    ax_side.set_ylim(min(z) - 50, max(z) + 50)
    ax_side.set_aspect('equal')
    ax_side.grid(True)
    ax_side.set_title("Side View (X-Z Plane)")
    ax_side.set_xlabel("X Position")
    ax_side.set_ylabel("Z Position (Height)")
    
    # Draw interaction surface at z=0
    ax_side.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Top view
    ax_top.plot(x, y, 'o-', lw=2)
    ax_top.plot([0, 25 * np.cos(base_rotation)], [0, 25 * np.sin(base_rotation)], 'r--', lw=1)
    ax_top.set_xlim(-300, 300)
    ax_top.set_ylim(-300, 300)
    ax_top.set_aspect('equal')
    ax_top.grid(True)
    ax_top.set_title("Top View (X-Y Plane)")
    ax_top.set_xlabel("X Position")
    ax_top.set_ylabel("Y Position")

def send_to_arduino(base_rotation, servo_theta1, servo_theta2, servo_theta3, serial_port='/dev/ttyACM0', baud_rate=9600):
    try:
        # Convert angles to integers by rounding
        base_deg = int(round(np.degrees(base_rotation)))
        theta1_deg = int(round(np.degrees(servo_theta1)))
        theta2_deg = int(round(np.degrees(servo_theta2))+360)
        theta3_deg = int(round(np.degrees(servo_theta3))+360)
        
        # Create the byte string
        command = b'%d,%d,%d,%d\n' % (base_deg, theta1_deg, theta2_deg, theta3_deg)
        print(f"\nSending command: {command}")
        
        # Initialize serial connection
        with serial.Serial(serial_port, baud_rate, timeout=2) as ser:
            # Clear any existing data
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            
            # Wait for Arduino to reset
            time.sleep(2)
            
            # Clear any startup messages
            while ser.in_waiting:
                ser.readline()
            
            # Send the command
            ser.write(command)
            ser.flush()
            
            # Wait for response
            response = ser.readline()
            
            if response:
                response_str = response.decode('utf-8').strip()
                print(f"Arduino response: '{response_str}'")
                return response_str == "OK"
            else:
                print("No response from Arduino")
                return False
                
    except serial.SerialException as e:
        print(f"\nSerial communication error: {e}")
        return False
    except Exception as e:
        print(f"\nError sending angles to Arduino: {e}")
        return False
    
class MainWindow(QMainWindow):
    def __init__(self, opencv_app):
        super().__init__()
        self.opencv_app = opencv_app
        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(30)
        
    def init_ui(self):
        self.setWindowTitle('Robot Arm Control System')
        self.setGeometry(50, 50, 1000, 800)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel for camera feed
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Camera feed display
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(600, 600)
        self.camera_label.setMaximumSize(600, 600)
        left_layout.addWidget(self.camera_label)
        
        # Button panel
        button_panel = QWidget()
        button_layout = QHBoxLayout(button_panel)

        # Create dropdown for target selection
        self.target_select = QComboBox()
        self.target_select.addItems(['Left Target (1)', 'Right Target (2)', 'Top Target (3)', 'Bottom Target (4)'])
        self.target_select.setStyleSheet('''
            QComboBox {
                background-color: #0d47a1;
                color: white;
                padding: 5px 15px;
                border: none;
                border-radius: 3px;
                min-width: 200px;
            }
            QComboBox:hover {
                background-color: #1565c0;
            }
            QComboBox QAbstractItemView {
                background-color: #1e1e1e;
                color: white;
                selection-background-color: #0d47a1;
            }
        ''')
        
        # Create buttons
        arduino_btn = QPushButton('Toggle Arduino (a)')
        quit_btn = QPushButton('Quit (q)')
        
        # Connect signals
        self.target_select.currentIndexChanged.connect(self.on_target_changed)
        arduino_btn.clicked.connect(self.on_arduino_clicked)
        quit_btn.clicked.connect(self.close)
        
        # Add widgets to layout
        button_layout.addWidget(self.target_select)
        button_layout.addWidget(arduino_btn)
        button_layout.addWidget(quit_btn)
        
        left_layout.addWidget(button_panel)
        
        # Right panel for arm visualization
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Arm visualization
        self.arm_label = QLabel()
        self.arm_label.setMinimumSize(500, 600)  # Adjust size to leave room for text
        self.arm_label.setMaximumSize(500, 600)
        right_layout.addWidget(self.arm_label)
        
        # Text output display
        self.text_display = QLabel()
        self.text_display.setMinimumSize(480, 200)  # Height for text display
        self.text_display.setStyleSheet("""
            QLabel {
                background-color: #1e1e1e;
                color: #00ff00;  /* Green text */
                font-family: 'Courier New';
                font-size: 12px;
                padding: 10px;
                border: 1px solid #3d3d3d;
            }
        """)
        self.text_display.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.text_display.setWordWrap(True)
        right_layout.addWidget(self.text_display)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        
        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        
        # Set style sheet
        self.setStyleSheet('''
            QMainWindow {
                background-color: #2b2b2b;
            }
            QLabel {
                background-color: #1e1e1e;
                border: 1px solid #3d3d3d;
            }
            QPushButton {
                background-color: #0d47a1;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 3px;
                min-width: 100px;
                margin: 2px;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QStatusBar {
                background-color: #1e1e1e;
                color: white;
            }
        ''')

    def keyPressEvent(self, event):
        # Handle number key presses for target selection
        if event.key() == Qt.Key_1:
            self.target_select.setCurrentIndex(0)  # Select Left Target
        elif event.key() == Qt.Key_2:
            self.target_select.setCurrentIndex(1)  # Select Right Target
        elif event.key() == Qt.Key_3:
            self.target_select.setCurrentIndex(2)  # Select Top Target
        elif event.key() == Qt.Key_4:
            self.target_select.setCurrentIndex(3)  # Select Bottom Target
        elif event.key() == Qt.Key_A:
            self.on_arduino_clicked()
        elif event.key() == Qt.Key_Q:
            self.close()

    def on_target_clicked(self, target_point):
        angles = self.opencv_app.move_to_current_target(target_point)
        if angles:
            # Display the calculated angles in the text display
            self.text_display.setText(
                f"<pre>\nTarget coordinates: {angles['target']}\n"
                f"Planar distance: {angles['planar_dist']:.2f}\n"
                f"Calculated joint positions for target: {angles['target']}\n"
                f"Base rotation: {np.degrees(angles['base_rotation']):.2f}°\n"
                f"Shoulder: {np.degrees(angles['servo_theta1']):.2f}°\n"
                f"Elbow: {np.degrees(angles['servo_theta2'])+360:.2f}°\n"
                f"Wrist: {np.degrees(angles['servo_theta3'])+360:.2f}°\n"
                     
            )

    def on_target_changed(self, index):
        # Map index to target names
        targets = ['left', 'right', 'top', 'bottom']
        self.on_target_clicked(targets[index])
        
    def update_display(self):
        try:
            # Update camera feed
            self.opencv_app.update_frame()
            
            if hasattr(self.opencv_app, 'processed_frame') and self.opencv_app.processed_frame is not None:
                # Make sure the frame is in the correct format
                frame = self.opencv_app.processed_frame.copy()  # Create a copy to ensure memory contiguity
                
                # Convert from BGR to RGB if needed
                if len(frame.shape) == 3:  # Color image
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                height, width = frame.shape[:2]
                bytes_per_line = 3 * width
                
                # Create QImage ensuring proper byte alignment
                q_image = QImage(frame.data,
                            width,
                            height,
                            bytes_per_line,
                            QImage.Format_RGB888)
                
                # Convert to pixmap and display
                pixmap = QPixmap.fromImage(q_image)
                self.camera_label.setPixmap(pixmap.scaled(
                    self.camera_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation))
                
            # Update arm visualization
            if hasattr(self.opencv_app, 'visualization_image') and self.opencv_app.visualization_image is not None:
                height, width = self.opencv_app.visualization_image.shape[:2]
                bytes_per_line = 3 * width
                q_image = QImage(self.opencv_app.visualization_image.data,
                            width,
                            height,
                            bytes_per_line,
                            QImage.Format_RGB888)
                self.arm_label.setPixmap(QPixmap.fromImage(q_image).scaled(
                    self.arm_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation))
                
        except Exception as e:
            print(f"Error in update_display: {e}")
    
    def on_contact_clicked(self):
        self.opencv_app.move_to_current_target()
    
    def on_arduino_clicked(self):
        self.opencv_app.toggle_arduino()
    
    def closeEvent(self, event):
        self.opencv_app.cleanup()
        event.accept()


class OpenCVApp:
    def __init__(self):
        # Arm parameters
        self.L1, self.L2, self.L3 = 130, 130, 75
        self.base_height = 0
        
        # Default arm position
        self.initial_base_rotation = np.radians(45)
        self.initial_theta1 = np.radians(160)
        self.initial_theta2 = np.radians(215)
        self.initial_theta3 = np.radians(210)
        
        # Grid parameters for camera feed
        self.grid_rows = 250
        self.grid_cols = 250
        self.latest_min_x_coord = None
        
        # Connect to Arduino flag
        self.arduino_connected = False
        
        # Visualization flags
        self.running = True
        self.status_message = "Ready"
        self.status_color = (0, 255, 0)  # Green
        
        # Attributes for Qt integration
        self.processed_frame = None
        self.visualization_image = None
        
        # Load the model
        print("Loading model...")
        try:
            # Try to find model in multiple possible locations
            model_paths = [
                '/Users/dylansomra/Desktop/IntegratedCV/Model/injurysegmentation.keras'
                
            ]
            
            for path in model_paths:
                try:
                    self.model = tf.keras.models.load_model(path, custom_objects={
                        'dice_coef': dice_coef,
                        'weighted_dice_coef': weighted_dice_coef,
                        'dice_coef_loss': dice_coef_loss,
                        'weighted_dice_coef_loss': weighted_dice_coef_loss,
                        'iou': iou,
                        'iou_loss': iou_loss,
                        'combined_dice_iou_loss': combined_dice_iou_loss
                    })
                    print(f"Model loaded successfully from {path}")
                    break
                except:
                    continue
            else:
                raise FileNotFoundError("Could not find model in any of the specified paths")
            
            # Setup initial arm visualization
            self.fig, (self.ax_side, self.ax_top) = plt.subplots(1, 2, figsize=(6, 6))
            self.update_arm_position(self.initial_base_rotation, 
                                   self.initial_theta1, 
                                   self.initial_theta2, 
                                   self.initial_theta3)
            
        except Exception as e:
            print(f"Error during initialization: {e}")
            self.status_message = f"Error: {str(e)}"
            self.status_color = (0, 0, 255)  # Red

    def update_frame(self):
        if not hasattr(self, 'cap') or not self.cap.isOpened():
            return False

        ret, frame = self.cap.read()
        if not ret:
            return False

        height, width = frame.shape[:2]
        size = min(height, width)
        start_x = (width - size) // 2
        start_y = (height - size) // 2
        frame = frame[start_y:start_y+size, start_x:start_x+size]
        
        # Resize to desired square size (e.g., 400x400)
        frame = cv2.resize(frame, (600, 600))
        # Process the frame for object detection
        contours, self.processed_frame = process_frame(frame, self.model)
        
        # Find the largest contour if any are detected
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get image dimensions and calculate grid size
            height, width = self.processed_frame.shape[:2]
            grid_size_x = width / self.grid_cols
            grid_size_y = height / self.grid_rows
            
            # Find extreme points
            left = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
            right = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])
            top = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
            bottom = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])
            
            # Convert all points to grid coordinates
            left_coord = (int(left[0] / grid_size_x), self.grid_rows - int(left[1] / grid_size_y))
            right_coord = (int(right[0] / grid_size_x), self.grid_rows - int(right[1] / grid_size_y))
            top_coord = (int(top[0] / grid_size_x), self.grid_rows - int(top[1] / grid_size_y))
            bottom_coord = (int(bottom[0] / grid_size_x), self.grid_rows - int(bottom[1] / grid_size_y))
            
            # Store all coordinates
            self.latest_coordinates = {
                'left': left_coord,
                'right': right_coord,
                'top': top_coord,
                'bottom': bottom_coord
            }
            
            # Draw contour and points on the frame
            cv2.drawContours(self.processed_frame, [largest_contour], -1, (0, 255, 0), 2)
            
            # Draw all points with labels
            cv2.circle(self.processed_frame, left, 5, (0, 255, 0), -1)
            cv2.circle(self.processed_frame, right, 5, (0, 0, 255), -1)
            cv2.circle(self.processed_frame, top, 5, (255, 0, 0), -1)
            cv2.circle(self.processed_frame, bottom, 5, (255, 255, 0), -1)
            
            # Draw coordinates for all points
            cv2.putText(self.processed_frame, f"Left: {left_coord}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(self.processed_frame, f"Right: {right_coord}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            cv2.putText(self.processed_frame, f"Top: {top_coord}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
            cv2.putText(self.processed_frame, f"Bottom: {bottom_coord}", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        return True
    
    def update_arm_position(self, base_rotation, theta1, theta2, theta3):
        # Calculate forward kinematics
        x, y, z = forward_kinematics(base_rotation, theta1, theta2, theta3, 
                                   self.L1, self.L2, self.L3, self.base_height)
        
        # Update plots
        plot_arm(self.ax_side, self.ax_top, x, y, z, self.base_height, base_rotation)
        
        # Save figure to memory buffer for Qt
        self.fig.canvas.draw()
        img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        self.visualization_image = img  # Already in RGB format
    
    def toggle_arduino(self):
        if not self.arduino_connected:
            try:
                # Test connection
                with serial.Serial('/dev/ttyACM0', 9600, timeout=1) as ser:
                    time.sleep(2)
                    self.status_message = "Arduino connected successfully"
                    self.status_color = (0, 255, 0)  # Green
                    self.arduino_connected = True
                    
                    # Send current position to Arduino
                    arduino_success = send_to_arduino(
                        self.initial_base_rotation,
                        self.initial_theta1,
                        self.initial_theta2,
                        self.initial_theta3
                    )
                    if not arduino_success:
                        self.status_message = "Connected but failed to send initial position"
                        self.status_color = (0, 165, 255)  # Orange
            except serial.SerialException as e:
                self.status_message = f"Arduino connection failed: {str(e)}"
                self.status_color = (0, 0, 255)  # Red
                self.arduino_connected = False
        else:
            self.status_message = "Arduino disconnected"
            self.status_color = (255, 0, 0)  # Blue
            self.arduino_connected = False
    
    def move_to_current_target(self, target_point='left'):
        if hasattr(self, 'latest_coordinates'):
            coords = self.latest_coordinates.get(target_point)
            if coords:
                target_x, target_y = coords
                target_z = 0
                
                print(f"Target {target_point} coordinates: ({target_x}, {target_y}, {target_z})")
                
                success, base_rotation, servo_theta1, servo_theta2, servo_theta3 = new_inverse_kinematics_3dof(
                    target_x, target_y, target_z, self.L1, self.L2, self.L3, self.base_height, 0
                )
                
                if success:
                    # Update visualization
                    self.update_arm_position(base_rotation, servo_theta1, servo_theta2, servo_theta3)
                    
                    # Calculate angles
                    base_int = round(np.degrees(base_rotation))
                    shoulder_int = round(np.degrees(servo_theta1))
                    elbow_int = round(np.degrees(servo_theta2)+360)
                    wrist_int = round(np.degrees(servo_theta3)+360)
                    
                    solution_text = f"{target_point.title()} Point - B:{base_int} S:{shoulder_int} E:{elbow_int} W:{wrist_int}"
                    self.status_message = solution_text
                    self.status_color = (255, 255, 0)  # Cyan
                    
                    # Send to Arduino if connected
                    if self.arduino_connected:
                        arduino_success = send_to_arduino(base_rotation, servo_theta1, servo_theta2, servo_theta3)
                        if arduino_success:
                            self.status_message = f"{solution_text} - Sent to Arduino"
                            self.status_color = (0, 255, 0)  # Green
                        else:
                            self.status_message = f"{solution_text} - Arduino communication failed"
                            self.status_color = (0, 0, 255)  # Red
                    
                    # Return calculated angles
                    return {
                    'base_rotation': base_rotation,
                    'servo_theta1': servo_theta1,
                    'servo_theta2': servo_theta2,
                    'servo_theta3': servo_theta3,
                    'target': (target_x, target_y, target_z),
                    'planar_dist': np.sqrt(target_x**2 + target_y**2)
                }
                else:
                    self.status_message = f"Inverse kinematics failed for {target_point} point"
                    self.status_color = (0, 0, 255)  # Red
                    return None
            else:
                self.status_message = f"No {target_point} point detected"
                self.status_color = (0, 0, 255)  # Red
                return None
        else:
            self.status_message = "No target detected"
            self.status_color = (0, 0, 255)  # Red
            return None
    
    def cleanup(self):
        """Cleanup method for proper shutdown"""
        self.running = False
        if hasattr(self, 'cap'):
            self.cap.release()
        plt.close('all')

def main():
    app = QApplication(sys.argv)
    opencv_app = OpenCVApp()  # Your existing OpenCV app
    
    # Modify the OpenCV app's run method to work with Qt
    opencv_app.cap = cv2.VideoCapture(0)
    if not opencv_app.cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Create and show the main window
    main_window = MainWindow(opencv_app)
    main_window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
