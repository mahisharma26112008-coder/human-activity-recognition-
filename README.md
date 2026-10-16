# human-activity-recognition-

1. Introduction
Human Activity Recognition (HAR) is a Deep Learning project used to identify and classify human activities such as walking, running, sitting, standing, and climbing stairs using sensor data or video images.
It uses machine learning and deep learning techniques to analyze data and predict which activity a person is performing.
2. Objective
The main objective of this project is to automatically recognize human activities using a deep learning model and improve activity monitoring in daily life.
3. Technologies Used
- Programming Language: Python
- Platform: Google Colab / Jupyter Notebook
- Libraries: NumPy, Pandas, TensorFlow/Keras, Matplotlib
- Model: CNN (Convolutional Neural Network)
- Dataset: Human activity dataset, such as UCI HAR
4. Working Principle
The project works through the following steps:
1. Data Collection: Collect sensor data or human activity images/videos.
2. Data Preprocessing: Clean and prepare the data for the model.
3. Model Training: Train the deep learning model using labeled activity data.
4. Activity Prediction: The model analyzes new data and predicts the activity.
5. Result Display: Display the predicted activity, such as walking or sitting.
5. Activities Recognized
- Walking
- Walking Upstairs
- Walking Downstairs
- Sitting
- Standing
- Laying Down
6. Applications
- Fitness and health monitoring
- Smartwatches and wearable devices
- Elderly care systems
- Sports activity tracking
- Smart home automation
- Human movement analysis
7. Advantages
- Automatically recognizes human activities.
- Reduces the need for manual monitoring.
- Supports real-time activity tracking when suitable hardware and models are used.
- Can be integrated with wearable devices.
8. Limitations
- Accuracy depends on the quality of the dataset.
- Different people may perform the same activity differently.
- Sensor placement and movement variations can affect predictions.
- Training may require significant computing resources for large datasets.
9. Conclusion
Human Activity Recognition using Deep Learning is a useful project that combines artificial intelligence and sensor data analysis to identify human activities automatically. It can be used in fitness monitoring, healthcare support, sports, and smart devices.

"""
Human Activity Recognition (HAR) with a 1D CNN + LSTM
Dataset: UCI HAR Dataset (smartphone accelerometer + gyroscope)
Download: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
Unzip so that the folder "UCI HAR Dataset" sits next to this script.

Install: pip install numpy pandas scikit-learn tensorflow matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical

DATA_DIR = "UCI HAR Dataset"
SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]
LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
          "SITTING", "STANDING", "LAYING"]


def load_split(split):
    """Return X of shape (samples, 128 timesteps, 9 channels) and labels (0-5)."""
    channels = []
    for sig in SIGNALS:
        path = f"{DATA_DIR}/{split}/Inertial Signals/{sig}_{split}.txt"
        channels.append(np.loadtxt(path))
    X = np.stack(channels, axis=-1)
    y = np.loadtxt(f"{DATA_DIR}/{split}/y_{split}.txt", dtype=int) - 1
    return X, y


def build_model(timesteps, n_features, n_classes):
    model = models.Sequential([
        layers.Input(shape=(timesteps, n_features)),
        layers.Conv1D(64, 3, activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.LSTM(64),
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def main():
    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")
    print("Train:", X_train.shape, "Test:", X_test.shape)

    # Standardize per channel using training statistics
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    y_train_cat = to_categorical(y_train, len(LABELS))
    y_test_cat = to_categorical(y_test, len(LABELS))

    model = build_model(X_train.shape[1], X_train.shape[2], len(LABELS))
    model.summary()

    history = model.fit(
        X_train, y_train_cat,
        validation_split=0.2,
        epochs=30,
        batch_size=64,
        callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=2,
    )

    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\nTest accuracy: {acc:.4f}")

    y_pred = model.predict(X_test).argmax(axis=1)
    print(classification_report(y_test, y_pred, target_names=LABELS))

    # Training curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Accuracy"); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss"); plt.legend()
    plt.tight_layout()
    plt.savefig("training_curves.png")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
    plt.yticks(range(len(LABELS)), LABELS)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.title("Confusion Matrix"); plt.tight_layout()
    plt.savefig("confusion_matrix.png")

    model.save("har_model.keras")


if __name__ == "__main__":
    main()
