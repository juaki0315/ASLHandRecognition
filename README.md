# **ASL Finger Spelling Recognition using YOLO and Deep Learning**

This project focuses on translating American Sign Language (ASL) finger spelling into text. The system detects hand keypoints from video frames, extracts features, and classifies each frame into corresponding letters of the alphabet.

## **🚀 Features**  
✅ Extracts **21 keypoints** per frame using **MediaPipe**.  
✅ Trains a **Neural Network (MLP)** to classify letters from keypoints.  
✅ Automatically saves the best model during training.  
✅ Can process video streams to generate ASL text output.  

## **🛠️ Installation**  
To set up the project, install the required dependencies:  
```bash
pip install mediapipe opencv-python numpy tensorflow scikit-learn
```

## **💡 Usage**  
1. **Train the Model:**  
   ```bash
   python train_model.py
   ```
2. **Run ASL Detection on Video:**  
   ```bash
   python detect_asl.py --video input.mp4
   ```

## **📂 Dataset**  
The dataset consists of images organized into folders, where each folder represents a letter (A-Z). The model learns from extracted keypoints instead of raw images, improving efficiency.  

## **📈 Model Architecture**  
- **Input:** 63 features (21 keypoints × 3 coordinates)  
- **Hidden Layers:** 128 → 64 neurons (ReLU activation)  
- **Output:** 26 classes (Softmax for A-Z classification)  

## **🔍 Future Improvements**  
- Enhance accuracy with **LSTMs for sequence prediction**.  
- Extend detection to **full ASL words and phrases**.  
- Deploy as a **real-time application**.  

