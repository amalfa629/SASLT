import tensorflow as tf
import numpy as np
import mediapipe as mp
import cv2
from tensorflow.keras.callbacks import ModelCheckpoint
from threading import Thread
import pyttsx3
from model.asl_classifier import load_model
from utils.landmarks import get_landmarks
from utils.dictionary import reverse_dictionary
from utils.VideoStream import VideoStream

"""
Forces code to use GPU acceleration when available
"""
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0], 'GPU')

"""
Handles text to speech threading.
"""
def speak(engine, currentWord):
    engine.say(currentWord)
    engine.runAndWait()


def main():
    model = load_model("weights/asl_classifier.keras")

    subStrings = [] #Keeps track of previously signed words
    currentWord = ""

    stream = VideoStream()
    frame_n = 0 #Used for frame parsing
    speed = 600 #Samples every n frames, starts at 600 but is variable

    #Sets up window environment
    black_background = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cv2.namedWindow("Synchronous American Sign Language Translator", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Synchronous American Sign Language Translator", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    #TTS engine
    engine = pyttsx3.init("espeak")
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1)

    while True:
        frame = stream.read()
        if frame_n % speed == 0:
            points = get_landmarks(frame)
            if not np.all(points == 0):
                test_image_landmarks = np.expand_dims(points, axis=0)
                predictions = model.predict(test_image_landmarks)
                prediction = np.argmax(predictions, axis=1)[0]
                char = reverse_dictionary.get(prediction)

                #Handles non-letter signs
                if char == "space":
                    subStrings.append(currentWord)
                    thread = Thread(target=speak, args=(engine, currentWord,))
                    thread.start()
                    currentWord = ""
                elif char == "del":
                    if len(currentWord) == 0 and len(subStrings) > 0:
                        currentWord = subStrings[-1]
                        subStrings = subStrings[:-1]
                    else:
                        currentWord = currentWord[:-1]
                elif len(char) == 1:    #Handles letter sign
                    currentWord += char

        if frame is None:
            continue

        #Keeps text in frame, and auto clears after a certain threshold.
        text = " ".join(subStrings + [currentWord])
        if len(text) > 70:
            subStrings = []
        if len(" ".join(subStrings)) > 30:
            midpoint = len(subStrings) // 2
            line1 = "Prediction: " + " ".join(subStrings[:midpoint])
            line2 = " ".join(subStrings[midpoint:] + [currentWord])
            lines = [line1, line2]
        else:
            line1 = "Prediction: " + text
            lines = [line1]

        #Caption bar
        font_scale = 0.6
        font_thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        line_spacing = 10
        x = 30
        frame = cv2.flip(frame, 1)
        y_start = frame.shape[0] - 30
        for i, line in enumerate(reversed(lines)):
            text_size, _ = cv2.getTextSize(line, font, font_scale, font_thickness)
            text_w, text_h = text_size
            y = y_start - i * (text_h + line_spacing)
            cv2.rectangle(frame, (x - 5, y - text_h - 5), (x + text_w + 5, y + 5), (0, 0, 0), -1)
            cv2.putText(frame, line, (x, y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        #Sizes camera feed
        fh, fw = frame.shape[:2]
        scale = min(1920 / fw, 1080 / fh)
        new_w, new_h = int(fw * scale), int(fh * scale)
        resized_frame = cv2.resize(frame, (new_w, new_h))
        background = black_background.copy()
        x_offset = (1920 - new_w) // 2
        y_offset = (1080 - new_h) // 2
        background[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_frame
        frame_n += 1
        cv2.imshow("Synchronous American Sign Language Translator", background)

        #Handles keyboard and sign commands
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or "quit" in subStrings:
            break
        elif key == ord('c') or "clc" in subStrings:
            subStrings = []
            currentWord = ""
        elif key == ord('w'):
            speed = 150
        elif key == ord('e'):
            speed = 300
        elif key == ord('r'):
            speed = 600

    stream.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()