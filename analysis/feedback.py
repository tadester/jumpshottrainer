import cv2
import mediapipe as mp
import numpy as np
import os

# Calculate the angle (in degrees) at point b given three points a, b, and c.
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Generate feedback based on the measured elbow angle.
def generate_feedback(elbow_angle, ideal_lower=80, ideal_upper=100):
    if elbow_angle < ideal_lower:
        return "Your elbow is too bent. Extend your arm."
    elif elbow_angle > ideal_upper:
        return "Your elbow is too straight. Bend your elbow more."
    else:
        return "Your elbow angle is ideal."

# Overlay feedback text on the image at a specified position.
def annotate_image_with_feedback(image, text, position=(10, 30)):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return image

# Process a single image: perform pose detection using MediaPipe, calculate the elbow angle,
# generate feedback, and annotate the image if the detected pose meets the filtering condition.
def process_image(image_path, threshold=0.5):
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        mp_drawing = mp.solutions.drawing_utils

        # Read the image from disk.
        image = cv2.imread(image_path)
        if image is None:
            return None

        # Convert the image to RGB format for processing.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # Continue only if pose landmarks are detected.
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Use the right wrist landmark (index 16) to determine if the person is on the right side.
            right_wrist_landmark = landmarks[16]
            if right_wrist_landmark.x > threshold:
                image_height, image_width, _ = image.shape
                # Convert selected landmarks to pixel coordinates.
                right_shoulder = (landmarks[12].x * image_width, landmarks[12].y * image_height)
                right_elbow = (landmarks[14].x * image_width, landmarks[14].y * image_height)
                right_wrist = (landmarks[16].x * image_width, landmarks[16].y * image_height)

                # Compute the elbow angle.
                elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                feedback_text = generate_feedback(elbow_angle)

                # Draw pose landmarks on the image.
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                # Annotate image with feedback text and calculated angle.
                image = annotate_image_with_feedback(image, feedback_text, (10, 30))
                angle_text = f"Elbow: {elbow_angle:.1f} deg"
                image = annotate_image_with_feedback(image, angle_text, (10, 60))
                return image
            else:
                return None
        else:
            return None

# Main function to process all JPEG images in the frames directory.
def main():
    # Determine the project root directory.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # Define the input directory containing the frames.
    frames_dir = os.path.join(project_root, 'data', 'frames')
    # Define the output directory for the annotated images.
    output_dir = os.path.join(project_root, 'data', 'analysis_results')
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all JPEG files in the frames directory.
    for filename in os.listdir(frames_dir):
        if filename.lower().endswith('.jpg'):
            image_path = os.path.join(frames_dir, filename)
            # Process each image and retrieve annotated version if the criteria are met.
            annotated_image = process_image(image_path, threshold=0.5)
            if annotated_image is not None:
                # Save the annotated image to the output directory.
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, annotated_image)
            else:
                # Image is skipped if pose detection fails or does not meet the condition.
                pass

if __name__ == "__main__":
    main()
