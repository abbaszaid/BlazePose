import cv2
import mediapipe as mp
import time
import socket
import json
from collections import deque

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

UDP_IP = "127.0.0.1"
UDP_PORT = 5065
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

cap = cv2.VideoCapture(1)
prevTime = 0

y_14_buffer = deque(maxlen=30)
y_13_buffer = deque(maxlen=30)
y_32_buffer = deque(maxlen=30)
distance_right_buffer = deque(maxlen=30)
distance_left_buffer = deque(maxlen=30)

def is_valid_landmark(x, y, z, visibility, image_width, image_height, visibility_threshold=0.5):
    valid_x = 0 <= x <= image_width
    valid_y = 0 <= y <= image_height
    valid_z = -2 * image_width <= z <= 2 * image_width
    valid_visibility = visibility >= visibility_threshold
    return valid_x and valid_y and valid_z and valid_visibility

def euclidean_distance(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image_height, image_width, _ = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        detected_pose = None

        if results.pose_landmarks:
            keypoints = []
            left_shoulder_x = left_shoulder_y = right_shoulder_x = right_shoulder_y = None
            left_wrist_x = left_wrist_y = right_wrist_x = right_wrist_y = None
            
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                x = landmark.x * image_width
                y = landmark.y * image_height
                z = landmark.z * image_width
                visibility = landmark.visibility

                if idx == 14:
                    if is_valid_landmark(x, y, z, visibility, image_width, image_height):
                        keypoints.append({"x": x, "y": 1 - y, "z": z})
                        y_14_buffer.append(y)
                        if len(y_14_buffer) == y_14_buffer.maxlen:
                            if (y_14_buffer[0] - y_14_buffer[-1]) > 100:
                                detected_pose = "Raising right hand"
                                y_14_buffer.clear()

                if idx == 13:
                    if is_valid_landmark(x, y, z, visibility, image_width, image_height):
                        keypoints.append({"x": x, "y": 1 - y, "z": z})
                        y_13_buffer.append(y)
                        if len(y_13_buffer) == y_13_buffer.maxlen:
                            if (y_13_buffer[0] - y_13_buffer[-1]) > 100:
                                detected_pose = "Raising left hand"
                                y_13_buffer.clear()

                if idx == 32:
                    if is_valid_landmark(x, y, z, visibility, image_width, image_height):
                        y_32_buffer.append(y)
                        if len(y_32_buffer) == y_32_buffer.maxlen:
                            if (y_32_buffer[0] - y_32_buffer[-1]) > 100:
                                detected_pose = "Raising right feet"
                                y_32_buffer.clear()

                if idx == 20:
                    if is_valid_landmark(x, y, z, visibility, image_width, image_height):
                        right_index_finger = landmark
                        right_ear = results.pose_landmarks.landmark[8]
                        distance_right = euclidean_distance(
                            right_index_finger.x * image_width, right_index_finger.y * image_height,
                            right_ear.x * image_width, right_ear.y * image_height
                        )
                        distance_right_buffer.append(distance_right)

                if idx == 19:
                    if is_valid_landmark(x, y, z, visibility, image_width, image_height):
                        left_index_finger = landmark
                        left_ear = results.pose_landmarks.landmark[7]
                        distance_left = euclidean_distance(
                            left_index_finger.x * image_width, left_index_finger.y * image_height,
                            left_ear.x * image_width, left_ear.y * image_height
                        )
                        distance_left_buffer.append(distance_left)

                if idx == 12:  # Left Shoulder
                    left_shoulder_x = x
                    left_shoulder_y = y
                if idx == 13:  # Right Shoulder
                    right_shoulder_x = x
                    right_shoulder_y = y
                if idx == 16:  # Left Wrist
                    left_wrist_x = x
                    left_wrist_y = y
                if idx == 17:  # Right Wrist
                    right_wrist_x = x
                    right_wrist_y = y

            # T-Pose detection logic
            if (
                left_wrist_x is not None and right_wrist_x is not None and
                left_shoulder_x is not None and right_shoulder_x is not None and
                abs(left_wrist_y - left_shoulder_y) < 40 and abs(right_wrist_y - right_shoulder_y) < 40 and
                abs(left_wrist_x - left_shoulder_x) > 30 and abs(right_wrist_x - right_shoulder_x) > 30
            ):
                detected_pose = "T-Pose"

            if distance_right_buffer and distance_left_buffer:
                if len(distance_right_buffer) == distance_right_buffer.maxlen and len(distance_left_buffer) == distance_left_buffer.maxlen:
                    if min(distance_right_buffer) < 50 and min(distance_left_buffer) < 50:
                        detected_pose = "Hear no evil pose"
                        distance_right_buffer.clear()
                        distance_left_buffer.clear()

            if detected_pose:
                print(detected_pose)
                message = json.dumps({"pose": detected_pose})
                sock.sendto(message.encode(), (UDP_IP, UDP_PORT))

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime
        cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)
        cv2.imshow('AbbasPose', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
sock.close()
