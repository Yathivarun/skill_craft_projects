import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Path to dataset folders
CAT_DIR = "C:/Users/pdx/OneDrive/Desktop/mini_project/z_internship/datasets/PetImages/Cat"
DOG_DIR = "C:/Users/pdx/OneDrive/Desktop/mini_project/z_internship/datasets/PetImages/Dog"

# Load and preprocess images
def load_images(folder, label, img_size=64, max_images=1000):
    data = []
    files = os.listdir(folder)[:max_images]
    for file in files:
        path = os.path.join(folder, file)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, (img_size, img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            data.append((img.flatten(), label))
    return data

# Load cat and dog images
cat_data = load_images(CAT_DIR, label=0)  # 0 = cat
dog_data = load_images(DOG_DIR, label=1)  # 1 = dog
data = cat_data + dog_data
np.random.shuffle(data)

# Split into X and y
X = np.array([x[0] for x in data])
y = np.array([x[1] for x in data])

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
print("Training SVM classifier...")
clf = SVC(kernel='linear')  # or 'rbf'
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1], ['Cat', 'Dog'])
plt.yticks([0, 1], ['Cat', 'Dog'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
plt.colorbar()
plt.tight_layout()
plt.show()

# -------- Predict on a custom image --------
print("\n--- Predict Your Own Image ---")
image_path = input("Enter the path to a cat/dog image: ")

try:
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_flat = img.flatten().reshape(1, -1)

    pred = clf.predict(img_flat)[0]
    label = "Dog 🐶" if pred == 1 else "Cat 🐱"
    print(f"Prediction: {label}")

    # Show image
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted: {label}")
    plt.axis('off')
    plt.show()

except Exception as e:
    print(f"Error: {e}")
