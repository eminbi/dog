import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd

def train_model(data_path):
    data = pd.read_csv(data_path)
    X = data.drop("label", axis=1)
    y = data["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    model.save("models/pet_behavior_model.h5")
