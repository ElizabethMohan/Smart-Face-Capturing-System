# Smart Face Capturing System

A computer vision-based real-time face capturing system using **OpenCV** and **MediaPipe**, designed to only capture images of **live human faces** after liveness verification through blink detection, face depth and facial tracking.

---

## 🚀 Features

- ✅ Detects **live and fake faces** using eye blink detection (EAR-based) and face depth.
- ✅ Warns when **multiple faces** are present.
- ✅ Aligns face to the center circle .
- ✅ Captures only when:
  - The **live face** is inside the capture frame.
  - User performs **long eye closure**.
- 📸 Displays and save captured image.

---

## 🧠 Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy

