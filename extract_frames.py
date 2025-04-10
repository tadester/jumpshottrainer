import cv2
import os
import numpy as np
from utils.preprocessing import preprocess_image

# Define paths relative to the project root
project_root = os.path.abspath(os.path.dirname(__file__))
raw_data_dir = os.path.join(project_root, 'data', 'raw')
frames_dir = os.path.join(project_root, 'data', 'frames')

os.makedirs(frames_dir, exist_ok=True)

def extract_and_preprocess_frames_from_video(video_path, output_dir, frame_interval=5, target_size=(128, 128)):
    """
    Extract frames from a video file, preprocess them, and overwrite with preprocessed images.
    
    :param video_path: Path to the video file.
    :param output_dir: Directory to save the frames.
    :param frame_interval: Process every n-th frame.
    :param target_size: Image size for preprocessing.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    # Get the video's base name for naming frames uniquely.
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    frame_count = 0
    saved_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save every frame_interval-th frame
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"{video_name}_frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved frame {saved_count} to {frame_filename}")
            
            # Preprocess the saved frame and overwrite the original file
            processed_image = preprocess_image(frame_filename, target_size=target_size)
            # Convert normalized image back to uint8 format for saving
            image_to_save = (processed_image * 255).astype('uint8')
            cv2.imwrite(frame_filename, image_to_save)
            print(f"Preprocessed and overwritten frame {saved_count} at {frame_filename}")
            
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted and preprocessed {saved_count} frames from {video_path}")

if __name__ == '__main__':
    # Process all video files in the raw data directory
    for filename in os.listdir(raw_data_dir):
        if filename.lower().endswith(('.mp4', '.avi', '.mov')):
            video_file = os.path.join(raw_data_dir, filename)
            extract_and_preprocess_frames_from_video(video_file, frames_dir, frame_interval=5, target_size=(128, 128))
