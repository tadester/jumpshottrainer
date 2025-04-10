import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Calculate the angle at point b given three points a, b, and c.
def calcangle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    ang = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(ang)

# Extract features (currently the elbow angle) from an image.
def extractfeat(imgpath):
    mppose = mp.solutions.pose
    with mppose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        img = cv2.imread(imgpath)
        if img is None:
            return None
        h, w, _ = img.shape
        imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = pose.process(imgrgb)
        if res.pose_landmarks is None:
            return None
        lands = res.pose_landmarks.landmark
        rshoulder = (lands[12].x * w, lands[12].y * h)
        relbow = (lands[14].x * w, lands[14].y * h)
        rwrist = (lands[16].x * w, lands[16].y * h)
        ang = calcangle(rshoulder, relbow, rwrist)
        return [ang]

# Load the dataset by reading labels from a CSV and extracting features from images.
def loaddata(framesdir, csvfile):
    df = pd.read_csv(csvfile)
    feats = []
    labs = []
    for idx, row in df.iterrows():
        fname = row['filename']
        lab = row['label']
        impath = os.path.join(framesdir, fname)
        feat = extractfeat(impath)
        if feat is not None:
            feats.append(feat)
            labs.append(lab)
    return np.array(feats), np.array(labs)

# Build a simple neural network classifier.
def buildmodel(inputdim):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_dim=inputdim),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    projroot = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    framesdir = os.path.join(projroot, 'data', 'frames')
    csvfile = os.path.join(projroot, 'data', 'annotations', 'shots.csv')
    X, y = loaddata(framesdir, csvfile)
    if len(X) == 0:
        print("No features extracted.")
        return
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
    mdl = buildmodel(Xtrain.shape[1])
    mdl.fit(Xtrain, ytrain, epochs=50, batch_size=8, validation_split=0.2)
    loss, acc = mdl.evaluate(Xtest, ytest)
    print("Test Accuracy:", acc)
    mdl.save(os.path.join(projroot, 'models', 'shotqualitymodel.keras'))

if __name__ == "__main__":
    main()
