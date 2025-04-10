import cv2
import mediapipe as mp
import os

def analyze_image(image_path, threshold=0.5):
    """
    Process a single image with MediaPipe Pose.
    If a pose is detected and the right wrist's normalized x-coordinate
    is above the specified threshold (indicating the person is on the right side),
    the function draws the landmarks and returns the annotated image.
    Otherwise, it returns None.
    
    Args:
        image_path (str): Path to the image file.
        threshold (float): Minimum normalized x value for the right wrist to annotate.
                           (0.0 is left edge, 1.0 is right edge)
    
    Returns:
        image (np.ndarray or None): Annotated image if conditions met, otherwise None.
    """
    mp_pose = mp.solutions.pose
    # It's best to use a context manager to handle initialization/cleanup.
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        mp_drawing = mp.solutions.drawing_utils

        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return None

        # Convert image from BGR to RGB for MediaPipe processing.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            # Get the right wrist (landmark index 16).
            right_wrist = landmarks[16]
            # Check if the right wrist x-coordinate is above the threshold.
            if right_wrist.x > threshold:
                # Draw landmarks since the person is assumed to be on the right side.
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                return image
            else:
                print(f"Skipping annotation for {image_path}: "
                      f"right wrist x value {right_wrist.x:.2f} is below threshold {threshold}")
                return None
        else:
            print(f"No pose detected for {image_path}")
            return None

def main():
    # Set up the paths. Since you're not using a separate preprocessed folder,
    # we will use the frames directory.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    frames_dir = os.path.join(project_root, 'data', 'frames')
    output_dir = os.path.join(project_root, 'data', 'analysis_results')
    os.makedirs(output_dir, exist_ok=True)

    # Process each image in the frames directory.
    for filename in os.listdir(frames_dir):
        if filename.lower().endswith('.jpg'):
            image_path = os.path.join(frames_dir, filename)
            # Adjust the threshold as needed (e.g., 0.5 means right half of the image).
            annotated_image = analyze_image(image_path, threshold=0.5)
            if annotated_image is not None:
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, annotated_image)
                print(f"Processed and saved: {output_path}")
            else:
                # Optionally, you can handle images that don't meet the condition.
                print(f"Image skipped (not annotated): {image_path}")

if __name__ == "__main__":
    main()
