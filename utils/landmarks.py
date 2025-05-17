import numpy as np
import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands   #Specifies we are using the hand tracking module of MediaPipe


"""
Introduces randomness to orientation of image in order to simulate left and right hands, as well as different hand angles.  Also zooms into random area of image.
This is all done to introduce variability into data to cover a wide range of hand positions that could mean the same letter.
"""
def augment_image(image):
    if np.random.rand() > 0.5:
        image = cv2.flip(image, 1)
    angle = np.random.randint(-30, 30)
    M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1)
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    zoom_factor = np.random.uniform(1, 1.2)
    height, width = image.shape[:2]
    new_width = int(width * zoom_factor)
    new_height = int(height * zoom_factor)
    image = cv2.resize(image, (new_width, new_height))
    image = image[int((new_height - height) / 2):int((new_height + height) / 2),
            int((new_width - width) / 2):int((new_width + width) / 2)]
    return image


"""
Scales down hand landmarks to uniform size of 1.00 by 1.00 in order to account for different hand sizes.
"""
def scale_points(points, xmin, ymin, xmax, ymax, x=1):
    width, height = xmax - xmin, ymax - ymin
    if width == 0 or height == 0:
        return [(0, 0)] * len(points)
    return [((p[0] - xmin) / width * x,
             (p[1] - ymin) / height * x) for p in points]


"""
Returns all twenty-one hand landmarks in a flattened area as [x0, y0, x1, y1... x20, y20], and returns an area of -1 if no landmarks are detected, which is used for error detection.
This is the only function called from outside of this context.
"""
def get_landmarks(frame):
    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5) as hands:
        image = augment_image(cv2.flip(frame, 1))
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            hand_landmarks = sorted(results.multi_hand_landmarks, key=lambda hl: hl.landmark[0].z)
            h, w, c = image.shape
            xList = []
            yList = []
            myHand = hand_landmarks[0].landmark
            hand_points = []
            for id, lm in enumerate(myHand):
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                hand_points.append([cx, cy])
            xmin, xmax = min(xList) - 20, max(xList) + 20
            ymin, ymax = min(yList) - 20, max(yList) + 20
            points = scale_points(hand_points, xmin, ymin, xmax, ymax)
            points = np.array(points)
            return points.flatten()
        else:
            print(f"No landmarks detected in image frame")
            return -1*np.ones(21 * 2)