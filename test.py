
import joblib
import numpy as np
import librosa

loaded_model = joblib.load('svm_voice_gender.pkl') 
scaler = joblib.load('scaler.joblib')  


audio_file = 'WhatsApp Audio 2024-11-15 at 10.26.55_d0256ee3.waptt.mp3'  
y, sr = librosa.load(audio_file, sr=None) 


mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
mfccs_mean = np.mean(mfccs, axis=1)  


mfccs_mean_scaled = scaler.transform(mfccs_mean.reshape(1, -1))


prediction = loaded_model.predict(mfccs_mean_scaled)


gender_label = "Male" if prediction[0] == 1 else "Female"
print(f"Predicted gender label: {gender_label}")
