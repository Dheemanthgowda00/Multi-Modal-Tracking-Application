import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
from typing import Dict, List, Tuple, Optional

class MultiTracker:
    def __init__(self):
        # Initialize MediaPipe solutions
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize tracking objects
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            max_num_hands=2
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Tracking modes
        self.modes = {
            'body': True,
            'hands': True,
            'eyes': True,
            'head': True
        }
        
        # Colors for different tracking elements
        self.colors = {
            'body': (0, 255, 0),      # Green
            'hands': (255, 0, 0),     # Blue
            'eyes': (0, 0, 255),      # Red
            'head': (255, 255, 0)     # Cyan
        }
        
        # Initialize pygame for audio feedback
        pygame.mixer.init()
        
        # Tracking data storage
        self.tracking_data = {
            'body_landmarks': None,
            'hand_landmarks': [],
            'eye_landmarks': None,
            'head_pose': None
        }
        
    def toggle_mode(self, mode: str):
        """Toggle a specific tracking mode on/off"""
        if mode in self.modes:
            self.modes[mode] = not self.modes[mode]
            print(f"{mode.capitalize()} tracking: {'ON' if self.modes[mode] else 'OFF'}")
    
    def process_body_tracking(self, frame, results):
        """Process and draw body pose tracking"""
        if not self.modes['body'] or not results.pose_landmarks:
            return frame
            
        # Draw pose landmarks
        self.mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        # Store body landmarks
        self.tracking_data['body_landmarks'] = results.pose_landmarks
        
        # Add body pose analysis
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            # Calculate body center
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
            body_center_x = int((left_hip.x + right_hip.x) * frame.shape[1] / 2)
            body_center_y = int((left_hip.y + right_hip.y) * frame.shape[0] / 2)
            
            # Draw body center
            cv2.circle(frame, (body_center_x, body_center_y), 10, self.colors['body'], -1)
            cv2.putText(frame, "BODY", (body_center_x + 15, body_center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['body'], 2)
        
        return frame
    
    def process_hand_tracking(self, frame, results):
        """Process and draw hand tracking"""
        if not self.modes['hands'] or not results.multi_hand_landmarks:
            return frame
            
        self.tracking_data['hand_landmarks'] = []
        
        for hand_landmarks, hand_type in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Draw hand landmarks
            self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Store hand landmarks
            self.tracking_data['hand_landmarks'].append({
                'landmarks': hand_landmarks,
                'type': hand_type.classification[0].label
            })
            
            # Add hand type label
            hand_type_text = hand_type.classification[0].label
            wrist = hand_landmarks.landmark[0]
            x, y = int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0])
            cv2.putText(frame, f"HAND: {hand_type_text}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['hands'], 2)
        
        return frame
    
    def process_eye_tracking(self, frame, results):
        """Process and draw eye tracking"""
        if not self.modes['eyes'] or not results.multi_face_landmarks:
            return frame
            
        self.tracking_data['eye_landmarks'] = results.multi_face_landmarks[0]
        
        for face_landmarks in results.multi_face_landmarks:
            # Draw face mesh
            self.mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            
            # Highlight eyes
            left_eye_center = self._get_eye_center(face_landmarks, 'left')
            right_eye_center = self._get_eye_center(face_landmarks, 'right')
            
            if left_eye_center:
                # Scale coordinates to actual frame dimensions
                left_eye_center = (
                    int(left_eye_center[0] * frame.shape[1] / 640),
                    int(left_eye_center[1] * frame.shape[0] / 480)
                )
                cv2.circle(frame, left_eye_center, 8, self.colors['eyes'], -1)
                cv2.putText(frame, "L_EYE", (left_eye_center[0] + 10, left_eye_center[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['eyes'], 1)
            
            if right_eye_center:
                # Scale coordinates to actual frame dimensions
                right_eye_center = (
                    int(right_eye_center[0] * frame.shape[1] / 640),
                    int(right_eye_center[1] * frame.shape[0] / 480)
                )
                cv2.circle(frame, right_eye_center, 8, self.colors['eyes'], -1)
                cv2.putText(frame, "R_EYE", (right_eye_center[0] + 10, right_eye_center[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['eyes'], 1)
        
        return frame
    
    def process_head_tracking(self, frame, results):
        """Process and draw head pose tracking"""
        if not self.modes['head'] or not results.multi_face_landmarks:
            return frame
            
        for face_landmarks in results.multi_face_landmarks:
            # Calculate head pose
            head_pose = self._calculate_head_pose(face_landmarks, frame.shape)
            self.tracking_data['head_pose'] = head_pose
            
            # Draw head pose indicators
            if head_pose:
                pitch, yaw, roll = head_pose
                
                # Draw head pose text
                cv2.putText(frame, f"PITCH: {pitch:.1f}°", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['head'], 2)
                cv2.putText(frame, f"YAW: {yaw:.1f}°", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['head'], 2)
                cv2.putText(frame, f"ROLL: {roll:.1f}°", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['head'], 2)
                
                # Draw head direction indicator
                center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
                cv2.circle(frame, (center_x, center_y), 50, self.colors['head'], 2)
                
                # Draw head direction line
                end_x = int(center_x + 100 * np.sin(np.radians(yaw)))
                end_y = int(center_y - 100 * np.sin(np.radians(pitch)))
                cv2.line(frame, (center_x, center_y), (end_x, end_y), self.colors['head'], 3)
        
        return frame
    
    def _get_eye_center(self, face_landmarks, eye_side: str):
        """Calculate the center of an eye based on landmarks"""
        if eye_side == 'left':
            eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        else:  # right eye
            eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        x_coords = [face_landmarks.landmark[i].x for i in eye_indices]
        y_coords = [face_landmarks.landmark[i].y for i in eye_indices]
        
        # Use standard resolution for calculations (will be scaled by the calling method)
        center_x = int(np.mean(x_coords) * 640)  # Standard width
        center_y = int(np.mean(y_coords) * 480)  # Standard height
        
        return (center_x, center_y)
    
    def _calculate_head_pose(self, face_landmarks, frame_shape):
        """Calculate head pose (pitch, yaw, roll) from face landmarks"""
        # Get key facial landmarks
        nose = face_landmarks.landmark[1]
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]
        left_ear = face_landmarks.landmark[234]
        right_ear = face_landmarks.landmark[454]
        
        # Calculate pitch (up/down)
        eye_center_y = (left_eye.y + right_eye.y) / 2
        pitch = (eye_center_y - nose.y) * 100
        
        # Calculate yaw (left/right)
        eye_center_x = (left_eye.x + right_eye.x) / 2
        yaw = (eye_center_x - nose.x) * 100
        
        # Calculate roll (tilt)
        ear_center_y = (left_ear.y + right_ear.y) / 2
        roll = (left_ear.y - right_ear.y) * 100
        
        return (pitch, yaw, roll)
    
    def process_frame(self, frame):
        """Process a single frame with all enabled tracking modes"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process each tracking mode
        if self.modes['body']:
            pose_results = self.pose.process(rgb_frame)
            frame = self.process_body_tracking(frame, pose_results)
        
        if self.modes['hands']:
            hands_results = self.hands.process(rgb_frame)
            frame = self.process_hand_tracking(frame, hands_results)
        
        if self.modes['eyes'] or self.modes['head']:
            face_results = self.face_mesh.process(rgb_frame)
            if self.modes['eyes']:
                frame = self.process_eye_tracking(frame, face_results)
            if self.modes['head']:
                frame = self.process_head_tracking(frame, face_results)
        
        # Add mode status display
        self._draw_mode_status(frame)
        
        return frame
    
    def _draw_mode_status(self, frame):
        """Draw current mode status on frame"""
        y_offset = 120
        for mode, is_active in self.modes.items():
            color = self.colors[mode] if is_active else (128, 128, 128)
            status = "ON" if is_active else "OFF"
            cv2.putText(frame, f"{mode.upper()}: {status}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 30
    
    def get_tracking_summary(self) -> Dict:
        """Get a summary of current tracking data"""
        summary = {
            'modes_active': {mode: status for mode, status in self.modes.items()},
            'body_detected': self.tracking_data['body_landmarks'] is not None,
            'hands_detected': len(self.tracking_data['hand_landmarks']),
            'eyes_detected': self.tracking_data['eye_landmarks'] is not None,
            'head_pose': self.tracking_data['head_pose']
        }
        return summary
    
    def cleanup(self):
        """Clean up resources"""
        self.pose.close()
        self.hands.close()
        self.face_mesh.close()

def main():
    """Main application loop"""
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Initialize tracker
    tracker = MultiTracker()
    
    print("Multi-Modal Tracking Application")
    print("Controls:")
    print("  B - Toggle body tracking")
    print("  H - Toggle hand tracking")
    print("  E - Toggle eye tracking")
    print("  T - Toggle head tracking")
    print("  Q - Quit")
    print("  S - Show tracking summary")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame with tracking
            processed_frame = tracker.process_frame(frame)
            
            # Display frame
            cv2.imshow('Multi-Modal Tracking', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('b'):
                tracker.toggle_mode('body')
            elif key == ord('h'):
                tracker.toggle_mode('hands')
            elif key == ord('e'):
                tracker.toggle_mode('eyes')
            elif key == ord('t'):
                tracker.toggle_mode('head')
            elif key == ord('s'):
                summary = tracker.get_tracking_summary()
                print("\n=== Tracking Summary ===")
                for key, value in summary.items():
                    print(f"{key}: {value}")
                print("=======================\n")
    
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    
    finally:
        # Cleanup
        tracker.cleanup()
        cap.release()
        cv2.destroyAllWindows()
        print("Application closed")

if __name__ == "__main__":
    main()
