import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def extract_features(image_path):
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        image = cv2.imread(image_path)
        if image is None:
            return None
        image_height, image_width, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if results.pose_landmarks is None:
            return None
        landmarks = results.pose_landmarks.landmark
        right_shoulder = (landmarks[12].x * image_width, landmarks[12].y * image_height)
        right_elbow = (landmarks[14].x * image_width, landmarks[14].y * image_height)
        right_wrist = (landmarks[16].x * image_width, landmarks[16].y * image_height)
        elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        return [elbow_angle]

def test_model(image_path, model_path):
    features = extract_features(image_path)
    if features is None:
        print("Feature extraction failed for", image_path)
        return
    features = np.array(features).reshape(1, -1)
    model = tf.keras.models.load_model(model_path)
    prediction = model.predict(features)[0][0]
    print("Predicted shot quality score (closer to 1 is good):", prediction)
    if prediction >= 0.5:
        print("Shot classified as GOOD")
    else:
        print("Shot classified as BAD")

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # Replace 'test_shot.jpg' with the filename of your test image in data/frames.
    test_image_path = os.path.join(project_root, 'data', 'frames', 'test_shot.jpg')
    model_path = os.path.join(project_root, 'models', 'shot_quality_model.h5')
    test_model(test_image_path, model_path)
