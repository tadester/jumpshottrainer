# Jump Shot Trainer

## Overview

Jump Shot Trainer is a sports performance analysis system designed to help basketball players and coaches evaluate and improve jump shot mechanics. The system leverages advanced pose estimation through MediaPipe and includes a custom-trained shot quality classifier built with TensorFlow. With functionality for frame extraction, pose analysis, feedback generation, and classifier training, the system provides both immediate biomechanical feedback and a data-driven evaluation of shot quality—all accessible through a user-friendly graphical interface.

## Features

- **Frame Extraction:**  
  Extract individual frames from raw video files for further analysis.

- **Pose Estimation and Feedback:**  
  Detect key body landmarks using MediaPipe. Calculate critical joint angles (e.g., the elbow angle) to provide immediate biomechanical feedback on jump shot mechanics.

- **Shot Quality Classifier:**  
  Train a custom classifier using labeled data to distinguish between “good” and “bad” jump shots. The classifier is built with TensorFlow and uses features extracted from video frames.

- **Evaluation:**  
  Evaluate individual images or videos by extracting features and using the trained classifier to generate a quality score and classification.

- **Graphical User Interface:**  
  An intuitive GUI provides buttons to run frame extraction, feedback analysis, video analysis, classifier training, and evaluation. Training logs and evaluation results are displayed in real time.

## Installation

### Prerequisites

- Python 3.7 or higher  
- Virtual environment (e.g., venv or conda)

### Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone <repository_url>
   cd jumpshottrainer
2. **Create and Activate a Virtual Environment**
python -m venv env
# For Windows:
env\Scripts\activate
# For macOS/Linux:
source env/bin/activate
3. **Install Dependencies**
Ensure that your requirements.txt includes:
opencv-python
numpy
pandas
mediapipe
scikit-learn
tensorflow
Then install all dependencies with:
pip install -r requirements.txt

**Graphical User Interface (GUI)**
Launch the GUI to access all functionalities (frame extraction, feedback analysis, video analysis, training, and evaluation) from a single control panel:
python gui.py


**Current Limitations and Future Enhancements**
# Current Limitations
Limited Feature Set:
Currently, the classifier uses only the elbow angle as a feature. This provides a baseline for shot quality evaluation.

Static Analysis:
Present analysis is performed on individual frames or a single frame from a video, rather than a complete temporal analysis of a shot sequence.

Basic Feedback:
Immediate feedback is based on preset thresholds, which may be improved by incorporating more nuanced biomechanical data.

# Future Enhancements
Expanded Feature Extraction:
Include additional features such as knee angles, shoulder alignment, and temporal dynamics from multiple frames.

Improved Classifier Architecture:
Experiment with more advanced neural network architectures or use LSTM-based models for handling time series data from videos.

Enhanced Video Analysis:
Develop functionality to process entire videos and provide continuous, real-time feedback.

Advanced GUI Features:
Enhance the GUI to support live camera feeds, detailed result visualizations, and interactive performance tracking.

Integration with Additional Sensors:
Consider incorporating data from wearable sensors to complement video analysis and provide a comprehensive evaluation.