import cv2
from threading import Thread


"""
Creates a separate thread for capturing the live video stream, this helps keep displayed video smooth and low latency
"""
class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()


    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()


    def read(self):
        return self.frame


    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()