import numpy as np
import os
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib

X, y = [], []

for file in os.listdir("data/landmarks"):
    label = file.split('_')[0]
    data = np.load(f"data/landmarks/{file}")
    X.extend(data)
    y.extend([label] * len(data))

X = np.array(X)
y = np.array(y)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

clf = SVC(kernel='rbf')
clf.fit(X, y_encoded)

joblib.dump(clf, 'models/svm_gesture_model.pkl')
joblib.dump(encoder, 'models/label_encoder.pkl')
print("Model trained and saved.")