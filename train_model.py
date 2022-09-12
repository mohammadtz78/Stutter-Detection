# This code is under MIT licence with accompanying file LICENSE.md

import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers
from keras import backend as K

parser = argparse.ArgumentParser(description='Train model.')
parser.add_argument('--dataset', type=str,
                    help='Path to dataset', default='dataset')

args = parser.parse_args()
dataset = args.dataset

IMAGE_COLORMODE = 'rgb'
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
BATCH_SIZE = 32
N_CHANNELS = 3
N_CLASSES = 2
CLASS_WEIGHT = {0: 1, 1: 3.693}  # Balance dataset
VALID_SPLIT = 0.1
EPOCHS = 50

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    validation_split=VALID_SPLIT,
    directory=dataset,
    shuffle=True,
    color_mode=IMAGE_COLORMODE,
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    subset="training",
    seed=7,
)

valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    validation_split=VALID_SPLIT,
    directory=dataset,
    shuffle=True,
    color_mode=IMAGE_COLORMODE,
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    subset="validation",
    seed=7,
)


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    pc = true_positives / (predicted_positives + K.epsilon())
    return pc


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    rc = true_positives / (possible_positives + K.epsilon())
    return rc


def f1(y_true, y_pred):
    pc = precision(y_true, y_pred)
    rc = recall(y_true, y_pred)
    return 2 * ((pc * rc) / (pc + rc + K.epsilon()))


# Create CNN model
model = tf.keras.models.Sequential([
    layers.Cropping2D(cropping=((59, 53), (81, 64))
                      ),  # Crop image from chart
    layers.Resizing(256, 256),  # Downscale image
    layers.Rescaling(1./255),  # Normalize image
    layers.Conv2D(32, 3, strides=2, padding='same', activation='swish'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.BatchNormalization(),
    layers.Conv2D(64, 3, padding='same', activation='swish'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.BatchNormalization(),
    layers.Conv2D(128, 3, padding='same', activation='swish'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(256, activation='swish'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(N_CLASSES, activation='softmax'),
])

# Compile model
model.compile(
    loss="categorical_crossentropy",
    optimizer='RMSprop',
    metrics=['accuracy', precision, recall, f1],
)

# Train model capture the history
history = model.fit(
    train_dataset, epochs=EPOCHS,
    validation_data=valid_dataset,
    class_weight=CLASS_WEIGHT,
)

# Compute the final loss, accuracy and etc.
final_loss, final_acc, final_precision, final_recall, f1_score = model.evaluate(
    valid_dataset, verbose=0)

print("Final loss: {0:.6f}, final accuracy: {1:.6f}, Final precision: {2:.6f}, Final recall: {3:.6f}, Final F1: {4:.6f}".format(
    final_loss,
    final_acc,
    final_precision,
    final_recall,
    f1_score,
))


def plot_metric(label, values, val_values):
    epochs = range(1, len(values) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, values, 'bo', label='Training ' + label)
    plt.plot(epochs, val_values, 'b', label='Validation ' + label)
    plt.title('Training and validation ' + label)
    plt.xlabel('Epochs')
    plt.ylabel(label)
    plt.legend()
    plt.savefig('plots/' + label + '.png')


# Plot the curves for training and validation.
history_dict = history.history
plot_metric('Loss', history_dict['loss'], history_dict['val_loss'])
plot_metric('Accuracy', history_dict['accuracy'], history_dict['val_accuracy'])
plot_metric(
    'Precision', history_dict['precision'], history_dict['val_precision'])
plot_metric('Recall', history_dict['recall'], history_dict['val_precision'])
plot_metric('F1', history_dict['f1'], history_dict['val_f1'])
