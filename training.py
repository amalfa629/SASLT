import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
from model.asl_classifier import ASLClassifier, load_model, save_model, train_model, evaluate_model
from utils.landmarks import get_landmarks
from utils.dictionary import dictionary

"""
Forces code to use GPU acceleration when available
"""
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0], 'GPU')

"""
Reads all images from training directory, compiling them into a TensorFlow Dataset in order to be efficient with memory usage.
"""
def create_dataset(dataset_directory, batch_size=32, shuffle_buffer_size=200000):
    image_count = 0
    image_points = []
    labels = []
    for folder_name in os.listdir(dataset_directory):
        folder_path = os.path.join(dataset_directory, folder_name)
        if os.path.isdir(folder_path):
            for image_name in os.listdir(folder_path):
                if image_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
                    image_path = os.path.join(folder_path, image_name)
                    a = get_landmarks(cv2.imread(image_path))
                    if not np.all(a == -1):
                        labels.append(dictionary[folder_name.lower()])
                        image_points.append(a)
                    print(image_count)
                    image_count += 1
    dataset = tf.data.Dataset.from_tensor_slices((image_points, labels))
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    total_size = len(image_points)
    train_size = int(total_size * 0.8)
    val_size = int(total_size * 0.2)
    train_dataset = dataset.take(train_size)
    validation_dataset = dataset.skip(train_size).take(val_size)
    train_dataset = train_dataset.shuffle(buffer_size=shuffle_buffer_size)
    validation_dataset = validation_dataset.shuffle(buffer_size=shuffle_buffer_size)
    train_dataset = train_dataset.batch(batch_size)
    validation_dataset = validation_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return train_dataset, validation_dataset


"""
Extracts batches from TensorFlow Dataset in order to save to a .npz file in the dataset directory.
"""
def extract_batched_features_labels(dataset):
    feature_list = []
    label_list = []
    for batch_features, batch_labels in dataset:
        feature_list.append(batch_features.numpy())
        label_list.append(batch_labels.numpy())
    all_features = np.concatenate(feature_list, axis=0)
    all_labels = np.concatenate(label_list, axis=0)
    return all_features, all_labels


def save_dataset(dataset, path):
    features, labels = extract_batched_features_labels(dataset)
    np.savez(path, features=features, labels=labels)


def load_dataset(path, batch_size = 32):
    data = np.load(path)
    features = data['features']
    labels = data['labels']
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.batch(batch_size)
    return dataset

"""
Plots data across all epochs of training using the values saved by the model's MetricsLogger.
"""
def plot_epoch_data(file = "metrics_log.csv"):
    df = pd.read_csv(file)
    plt.figure()
    plt.plot(df['epoch'], df['accuracy'], label='Train Accuracy')
    plt.plot(df['epoch'], df['val_accuracy'], label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.figure()
    plt.plot(df['epoch'], df['loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    #Used to toggle data scrapping, training, and evaluation settings.
    collect_dataset = False
    train = False
    evaluate = True

    #Model parameters, able to be easily altered
    input_size = 42
    hidden_size = 128
    output_size = 29
    epochs = 20

    model = ASLClassifier(input_size, hidden_size, output_size)

    if collect_dataset:
        dataset_directory = 'data/train'
        train_dataset, validation_dataset = create_dataset(dataset_directory, batch_size=32)
        save_dataset(train_dataset, 'dataset/train.npz')
        save_dataset(validation_dataset, 'dataset/valid.npz')

    train_dataset = load_dataset('dataset/train.npz')
    validation_dataset = load_dataset('dataset/valid.npz')

    if train:
        train_model(model, train_dataset, validation_dataset, epochs)
        save_model(model)

    model = load_model()

    if evaluate:
        evaluate_model(model, validation_dataset, dictionary)
        plot_epoch_data()


if __name__ == "__main__":
    main()
