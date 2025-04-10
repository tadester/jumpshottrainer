import tkinter as tk
from tkinter import filedialog
from tkinter.scrolledtext import ScrolledText
import subprocess
import sys
import threading
import queue
import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

def extract():
    subprocess.run([sys.executable, "extract_frames.py"])

def feedback():
    subprocess.run([sys.executable, "analysis/feedback.py"])

def video():
    subprocess.run([sys.executable, "analysis/video_analysis.py"])

def runcommand(cmd, outq):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    for line in proc.stdout:
        outq.put(line)
    proc.stdout.close()
    proc.wait()
    outq.put("TRAINING COMPLETE\n")

def update(widget, outq):
    try:
        while True:
            line = outq.get_nowait()
            widget.insert(tk.END, line)
            widget.see(tk.END)
    except queue.Empty:
        pass
    widget.after(100, update, widget, outq)

def train():
    outtext.delete("1.0", tk.END)
    cmd = [sys.executable, "models/shotquality.py"]
    t = threading.Thread(target=runcommand, args=(cmd, trainqueue))
    t.daemon = True
    t.start()

def extractfeatures(imgpath):
    mpPose = mp.solutions.pose
    with mpPose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        img = cv2.imread(imgpath)
        if img is None:
            return None
        h, w, _ = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        if results.pose_landmarks is None:
            return None
        lands = results.pose_landmarks.landmark
        rs = (lands[12].x * w, lands[12].y * h)
        re = (lands[14].x * w, lands[14].y * h)
        rw = (lands[16].x * w, lands[16].y * h)
        a, b, c = np.array(rs), np.array(re), np.array(rw)
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        ang = np.arccos(np.clip(cosine, -1.0, 1.0))
        return [np.degrees(ang)]

def evalimage():
    fpath = filedialog.askopenfilename(title="Select an image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png"), ("All Files", "*.*")])
    if not fpath:
        return
    feats = extractfeatures(fpath)
    if feats is None:
        outtext.insert(tk.END, f"Feature extraction failed for {fpath}\n")
        return
    feats = np.array(feats).reshape(1, -1)
    proj = os.path.join(os.path.dirname(__file__), "")
    modpath = os.path.join(proj, "models", "shotqualitymodel.keras")
    try:
        mdl = tf.keras.models.load_model(modpath)
    except Exception as e:
        outtext.insert(tk.END, f"Error loading model: {str(e)}\n")
        return
    pred = mdl.predict(feats)[0][0]
    cls = "GOOD" if pred >= 0.5 else "BAD"
    res = f"Shot Quality Score: {pred:.3f} - {cls}\n"
    outtext.insert(tk.END, res)
    outtext.see(tk.END)

def evalvideo():
    fpath = filedialog.askopenfilename(title="Select a video", filetypes=[("Video Files", "*.mp4;*.avi;*.mov"), ("All Files", "*.*")])
    if not fpath:
        return
    cap = cv2.VideoCapture(fpath)
    ret, frm = cap.read()
    cap.release()
    if not ret:
        outtext.insert(tk.END, "Failed to extract frame from video.\n")
        return
    temp = os.path.join(os.path.dirname(fpath), "temp_frame.jpg")
    cv2.imwrite(temp, frm)
    feats = extractfeatures(temp)
    os.remove(temp)
    if feats is None:
        outtext.insert(tk.END, "Feature extraction failed for video frame.\n")
        return
    feats = np.array(feats).reshape(1, -1)
    proj = os.path.join(os.path.dirname(__file__), "")
    modpath = os.path.join(proj, "models", "shotqualitymodel.keras")
    try:
        mdl = tf.keras.models.load_model(modpath)
    except Exception as e:
        outtext.insert(tk.END, f"Error loading model: {str(e)}\n")
        return
    pred = mdl.predict(feats)[0][0]
    cls = "GOOD" if pred >= 0.5 else "BAD"
    res = f"Video Frame Shot Quality Score: {pred:.3f} - {cls}\n"
    outtext.insert(tk.END, res)
    outtext.see(tk.END)

root = tk.Tk()
root.title("Jump Shot Trainer")
root.geometry("600x600")

framebuttons = tk.Frame(root)
framebuttons.pack(pady=10)

btnextract = tk.Button(framebuttons, text="Extract Frames", command=extract, width=25, height=2)
btnextract.grid(row=0, column=0, padx=5, pady=5)

btnfeedback = tk.Button(framebuttons, text="Run Feedback Analysis", command=feedback, width=25, height=2)
btnfeedback.grid(row=0, column=1, padx=5, pady=5)

btnvideo = tk.Button(framebuttons, text="Run Video Analysis", command=video, width=25, height=2)
btnvideo.grid(row=1, column=0, padx=5, pady=5)

btntrain = tk.Button(framebuttons, text="Train Classifier", command=train, width=25, height=2)
btntrain.grid(row=1, column=1, padx=5, pady=5)

btnevalimg = tk.Button(framebuttons, text="Evaluate Shot (Image)", command=evalimage, width=25, height=2)
btnevalimg.grid(row=2, column=0, padx=5, pady=5)

btnevalvid = tk.Button(framebuttons, text="Evaluate Shot (Video)", command=evalvideo, width=25, height=2)
btnevalvid.grid(row=2, column=1, padx=5, pady=5)

outtext = ScrolledText(root, width=70, height=20)
outtext.pack(pady=10)

trainqueue = queue.Queue()
update(outtext, trainqueue)

root.mainloop()
