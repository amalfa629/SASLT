import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import csv
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from utils.dictionary import reverse_dictionary
import matplotlib.pyplot as plt

"""
Resources for model, which include our learning rate scheduler, as well as a way to save the best trained model according to our validation set.
"""
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, verbose=1)
checkpoint_callback = ModelCheckpoint('weights/best_model.keras',
                                      monitor='val_accuracy',
                                      save_best_only=True,
                                      verbose=1)

"""
Model structure and built int functions, such as call.
"""
class ASLClassifier(tf.keras.Model):
    def __init__(self, input_size=42, hidden_size=128, output_size=29, **kwargs):
        super(ASLClassifier, self).__init__()
        self.fc1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.3)
        self.fc2 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        self.fc3 = tf.keras.layers.Dense(output_size, activation='softmax')


    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return self.fc3(x)


    def get_config(self):
        return {
            "input_size": 42,
            "hidden_size": 128,
            "output_size": 29
        }


    @classmethod
    def from_config(cls, config):
        return cls(**config)

"""
Used for saving data across each epoch to a csv file.
"""
class MetricsLogger(tf.keras.callbacks.Callback):
    def __init__(self, filename="metrics_log.csv"):
        super().__init__()
        self.filename = filename
        with open(self.filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "loss", "accuracy", "val_loss", "val_accuracy"])


    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch + 1,
                logs.get("loss"),
                logs.get("accuracy"),
                logs.get("val_loss"),
                logs.get("val_accuracy")
            ])


def train_model(model, train_dataset, validation_dataset, epochs=20):
    metrics_logger = MetricsLogger("../metrics_log.csv")
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_dataset, epochs=epochs,
              validation_data=validation_dataset,
              callbacks=[lr_scheduler, checkpoint_callback, metrics_logger])
    print("Training completed!")


def save_model(model, path="weights/asl_classifier.keras"):
    model.save(path)
    print("Model weights saved successfully!")


def load_model(path="weights/asl_classifier.keras"):
    model = tf.keras.models.load_model(path, custom_objects={"ASLClassifier": ASLClassifier})
    return model

"""
Evaluates model based on given dataset.
"""
def evaluate_model(model, dataset, class_names):
    loss, accuracy = model.evaluate(dataset)
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    y_true = []
    y_pred = []
    for x_batch, y_batch in dataset:
        preds = model.predict(x_batch)
        y_true.extend(y_batch.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true_named = [reverse_dictionary[i] for i in y_true]
    y_pred_named = [reverse_dictionary[i] for i in y_pred]
    print("\nClassification Report:")
    print(classification_report(y_true_named, y_pred_named, labels = list(reverse_dictionary.values())))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()