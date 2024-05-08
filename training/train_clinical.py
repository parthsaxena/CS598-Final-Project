import pandas as pd
import numpy as np
import os
import random
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    # Load the data
    X_train = pd.read_pickle("../preprocess_clinical/X_train_c.pkl")
    y_train = pd.read_pickle("../preprocess_clinical/y_train_c.pkl")
    X_test = pd.read_pickle("../preprocess_clinical/X_test_c.pkl")
    y_test = pd.read_pickle("../preprocess_clinical/y_test_c.pkl")

    # Adjust data types
    X_train = X_train.astype('float32')
    y_train = y_train.astype('int32')
    X_test = X_test.astype('float32')
    y_test = y_test.astype('int32')

    # Build the model
    model = Sequential()
    model.add(Input(shape=(113,)))  # Adjust the input shape to match the actual feature count
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(50, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(3, activation="softmax"))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])

    # Model summary
    model.summary()

    # Train the model
    history = model.fit(X_train, y_train, epochs=100, validation_split=0.1, batch_size=32, verbose=1)

    # Evaluate the model
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

    # Predictions
    test_predictions = model.predict(X_test)
    test_label = to_categorical(y_test, 3)
    true_label = np.argmax(test_label, axis=1)
    predicted_label = np.argmax(test_predictions, axis=1)
    cr = classification_report(true_label, predicted_label, output_dict=True)
    print("Classification Report:", cr)

if __name__ == '__main__':
    main()
