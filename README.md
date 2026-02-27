# Hand Gesture Recognition

A machine learning project that classifies hand gestures using MediaPipe landmarks extracted from the HaGRID dataset.

## Project Structure
```
ML1_Project/
│
├── data/
│   └── hand_landmarks.csv
│
├── models/
│   └── gesture_model.pkl
│
│
├── Emam_ML1
├── video.py
└── requirements.txt
```

## Dataset
HaGRID (Hand Gesture Recognition Image Dataset) — 18 gesture classes.  
Each sample contains 21 hand landmarks (x, y, z) extracted using MediaPipe.

## Preprocessing
- Recentered landmarks by subtracting the wrist (landmark 0)
- Scaled using Euclidean distance to middle fingertip (landmark 12)
- Features ordered as: x1 y1 z1 x2 y2 z2 ... x21 y21 z21

## Models Trained
| Model | Description |
|---|---|
| SVC | Support Vector Classifier with linear and RBF kernels |
| Logistic Regression | Multinomial logistic regression |
| Random Forest | Ensemble of decision trees |

Hyperparameter tuning done using GridSearchCV with holdout validation (PredefinedSplit). (80% - 10% - 10%)

## Results
| Model | Val Accuracy | Test Accuracy |
|---|---|---|
| SVC | - | - |
| Logistic Regression | - | - |
| Random Forest | - | - |

## Real-Time Demo
Run the webcam demo with the best model:
```bash
python video.py
```
Press `q` to quit. Output saved as `videos/output_video.mp4`.

## Installation
```bash
conda create -n gesture_env python=3.10
conda activate gesture_env
pip install -r requirements.txt
```

## Demo Video
[Google Drive Link](https://drive.google.com/file/d/1MZHkQt4mjfYIrlqPpCXU3NR_NY9B_tDf/view?usp=sharing)