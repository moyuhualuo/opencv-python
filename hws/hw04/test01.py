import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# Directory where the training images are stored
train_directory = r'D:\opencv\hws\hw04\archive\classified_train'

# Function to extract color features
def extract_color_features(image_path):
    img = cv2.imread(image_path)
    avg_color = np.mean(img, axis=(0, 1))  # Placeholder for color feature extraction
    return avg_color

# Function to extract shape features
def extract_shape_features(image_path):
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    aspect_ratio = width / height  # Placeholder for shape feature extraction
    return aspect_ratio

# Load and extract features from all images in the directory
def load_and_extract_features(directory):
    images = []
    labels = []

    fruit_types = os.listdir(directory)
    for fruit_type in fruit_types:
        fruit_dir = os.path.join(directory, fruit_type)
        for file_name in os.listdir(fruit_dir):
            if file_name.endswith('.jpg'):
                file_path = os.path.join(fruit_dir, file_name)
                color_feature = extract_color_features(file_path)
                shape_feature = extract_shape_features(file_path)

                # Assign weights to features
                weighted_feature = 0.8 * color_feature + 0.20 * shape_feature + color_feature ** 2

                images.append(weighted_feature)
                labels.append(fruit_type)

    return np.array(images), np.array(labels)

# Load and extract features
X, y = load_and_extract_features(train_directory)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a RandomForestClassifier
rf = RandomForestClassifier(random_state=43)

# Define parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(rf, param_grid, cv=3, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model from GridSearchCV
best_rf = grid_search.best_estimator_

# Evaluate accuracy on test data
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"RandomForestClassifier accuracy after optimization: {accuracy:.2f}")

# Print best parameters found by GridSearchCV
print(f"Best parameters: {grid_search.best_params_}")


# Predict on a single image
def function(address):
    image_path = address

    color_feature = extract_color_features(image_path)
    shape_feature = extract_shape_features(image_path)
    weighted_feature = 0.8 * color_feature + 0.20 * shape_feature + color_feature ** 2

    # Reshape to match the expected input shape for prediction
    weighted_feature_reshaped = weighted_feature.reshape(1, -1)

    # Predict using the model
    prediction = best_rf.predict(weighted_feature_reshaped)[0]
    return prediction