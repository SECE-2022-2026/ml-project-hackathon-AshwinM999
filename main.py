import numpy as np
import pandas as pd
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

def load_data(data_path):
    data = []
    labels = []
    for gender in ['male', 'female']:
        path = os.path.join(data_path, gender)
        label = 1 if gender == 'male' else 0
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if file_path.endswith('.mp3'):
                try:
                    audio, sr = librosa.load(file_path, sr=None)
                    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                    mfccs_mean = np.mean(mfccs.T, axis=0)
                    data.append(mfccs_mean)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    return np.array(data), np.array(labels)

data_path = "./split_dataset"  
X, y = load_data(data_path)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


joblib.dump(scaler, 'scaler.joblib')
joblib.dump(svm_model, 'svm_voice_gender.pkl')
