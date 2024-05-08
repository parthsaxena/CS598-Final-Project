import os
import random
import gc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization, Conv2D, MultiHeadAttention, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import seaborn as sns
from sklearn.metrics import classification_report, precision_recall_curve, precision_recall_fscore_support

# Set TensorFlow session for GPU memory management
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# Function to load and process image data from pickle files
def make_img(t_img):
    img = pd.read_pickle(t_img)
    img_l = [img.values[i][0] for i in range(len(img))]
    return np.array(img_l).reshape(-1, 72, 72, 3)

# Function to set random seeds for reproducibility
def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# Model for Clinical data
def create_model_clinical():
    model = tf.keras.Sequential([
        Dense(200, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(100, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(50, activation="relu"),
        BatchNormalization(),
        Dropout(0.2)
    ])
    return model

# Model for Image data
def create_model_img():
    model = tf.keras.Sequential([
        Conv2D(72, (3, 3), activation='relu', input_shape=(72, 72, 3)),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(32, (3, 3), activation='relu'),
        Flatten(),
        Dense(50, activation='relu')
    ])
    return model

# Function to handle cross-modal and self-attention mechanisms
def cross_modal_attention(x, y):
    x = tf.expand_dims(x, axis=1)
    y = tf.expand_dims(y, axis=1)
    a1 = MultiHeadAttention(num_heads=4, key_dim=50)(x, y)
    a2 = MultiHeadAttention(num_heads=4, key_dim=50)(y, x)
    return concatenate([a1[:, 0, :], a2[:, 0, :]])

def self_attention(x):
    x = tf.expand_dims(x, axis=1)
    attention = MultiHeadAttention(num_heads=4, key_dim=50)(x, x)
    return attention[:, 0, :]

# Function to build the multi-modal model with only clinical and image data
def multi_modal_model(mode, train_clinical, train_img):
    in_clinical = Input(shape=(train_clinical.shape[1],))
    in_img = Input(shape=(train_img.shape[1], train_img.shape[2], train_img.shape[3]))
    
    dense_clinical = create_model_clinical()(in_clinical)
    dense_img = create_model_img()(in_img)
    
    if mode == 'MM_SA':
        img_att = self_attention(dense_img)
        clinical_att = self_attention(dense_clinical)
        merged = concatenate([img_att, clinical_att, dense_img, dense_clinical])
    elif mode == 'None':
        merged = concatenate([dense_img, dense_clinical])
    else:
        print("Mode must be 'MM_SA' or 'None'.")
        return None
    
    output = Dense(3, activation='softmax')(merged)
    model = Model([in_clinical, in_img], output)        
    return model

# Main training function
def train(mode, batch_size, epochs, learning_rate, seed):
    train_clinical = pd.read_csv("pkl/X_train_clinical.csv").values
    test_clinical = pd.read_csv("pkl/X_test_clinical.csv").values
    train_img = make_img("pkl/X_train_img.pkl")
    test_img = make_img("pkl/X_test_img.pkl")
    train_label = pd.read_csv("pkl/y_train.csv").values.flatten()
    test_label = pd.read_csv("pkl/y_test.csv").values.flatten()

    reset_random_seeds(seed)
    class_weights = compute_class_weight('balanced', classes=np.unique(train_label), y=train_label)
    class_weights = dict(enumerate(class_weights))
    
    model = multi_modal_model(mode, train_clinical, train_img)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    
    history = model.fit([train_clinical, train_img], train_label, epochs=epochs, batch_size=batch_size, class_weight=class_weights, validation_split=0.1, verbose=1)
    
    score = model.evaluate([test_clinical, test_img], test_label, verbose=0)
    test_predictions = model.predict([test_clinical, test_img])
    cr = classification_report(test_label, np.argmax(test_predictions, axis=1), output_dict=True)
    
    print('Test Accuracy:', score[1])
    print(classification_report(test_label, np.argmax(test_predictions, axis=1)))

    # Cleanup to free memory
    K.clear_session()
    gc.collect()

# if __name__ == "__main__":
#     train('MM_SA', 32, 50, 0.001, 123)  # Example parameters

if __name__=="__main__":
    
    m_a = {}
    seeds = random.sample(range(1, 200), 5)
    for s in seeds:
        acc, bs_, lr_, e_ , seed= train('MM_SA_BA', 32, 50, 0.001, s)
        m_a[acc] = ('MM_SA_BA', acc, bs_, lr_, e_, seed)
    print(m_a)
    print ('-'*55)
    max_acc = max(m_a, key=float)
    print("Highest accuracy of: " + str(max_acc) + " with parameters: " + str(m_a[max_acc]))
    
