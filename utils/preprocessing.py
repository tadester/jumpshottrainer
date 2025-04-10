import cv2
import numpy as np
import os

def preprocess_image(image_path, target_size=(128, 128)):
    """
    Load an image from a given file path, resize it to the target size, and normalize pixel values.

    Args:
        image_path (str): The file path of the image to be processed.
        target_size (tuple): The desired image size (width, height).

    Returns:
        np.ndarray: The preprocessed image as a NumPy array with pixel values in the range [0, 1].
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}. Check the file path.")
    
    # Resize the image to the target size
    image_resized = cv2.resize(image, target_size)
    
    # Normalize pixel values from 0-255 to 0-1
    image_normalized = image_resized.astype('float32') / 255.0
    return image_normalized

def preprocess_all_images(frames_dir, output_dir, target_size=(128, 128)):
    """
    Preprocess all JPEG images in a given directory and save the processed images into another directory.

    Args:
        frames_dir (str): Path to the directory containing raw JPEG images.
        output_dir (str): Path to the directory where preprocessed images will be saved.
        target_size (tuple): Desired image size (width, height) for resizing.
    """
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(frames_dir):
        if filename.lower().endswith('.jpg'):
            input_path = os.path.join(frames_dir, filename)
            # Preprocess the image
            processed_image = preprocess_image(input_path, target_size=target_size)
            
            # Define the output path
            output_path = os.path.join(output_dir, filename)
            # Convert processed image to uint8 (0-255) for saving via OpenCV
            image_to_save = (processed_image * 255).astype('uint8')
            cv2.imwrite(output_path, image_to_save)
            print(f"Preprocessed image saved to {output_path}")

if __name__ == '__main__':
    # Determine the project root (assumes this file is in a subfolder, e.g., 'utils')
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Define the directories for raw frames and the output preprocessed images.
    frames_directory = os.path.join(project_root, 'data', 'frames')
    output_directory = os.path.join(project_root, 'data', 'preprocessed')
    
    print(f"Frames directory: {frames_directory}")
    print(f"Output directory: {output_directory}")
    
    # Process all images in the frames directory
    preprocess_all_images(frames_directory, output_directory)
