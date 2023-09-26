import cv2
import mediapipe as mp
import time
import math
import numpy as np
import drivers
import RPi.GPIO as GPIO
display = drivers.Lcd()
display.lcd_clear()

bar_repr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
class poseDetector():

    def __init__(self, static_image_mode: bool = False,
                 model_complexity: int = 1,
                 smooth_landmarks: bool = True,
                 enable_segmentation: bool = False,
                 smooth_segmentation: bool = True,
                 min_detection_confidence: float = 0.9,
                 min_tracking_confidence: float = 0.9):
        self.mode = static_image_mode
        self.complexity = model_complexity
        self.landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        self.pose = self.mpPose.Pose(
            self.mode, self.complexity, self.landmarks,
            self.enable_segmentation, self.smooth_segmentation,
            self.min_detection_confidence, self.min_tracking_confidence
        )

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):

        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        # print(angle)

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle


def main():
    display.lcd_display_string("AiFitnessTrainer", 1)
    time.sleep(2)
    def bicepCurls():

        cap = cv2.VideoCapture(0)
        detector = poseDetector()
        count = 0
        dir = 0
        while count < 20:
            success, img = cap.read()
                # img = cv2.imread("AiTrainer/test.jpg")
            img = detector.findPose(img, False)
            lmList = detector.findPosition(img, False)
                # print(lmList)
            if len(lmList) != 0:
                    # Right Arm
                if count % 2 == 0:
                    angle = detector.findAngle(img, 12, 14, 16)
                else:
                    angle = detector.findAngle(img, 15, 13, 11)
                    # # Left Arm
                    # angle = detector.findAngle(img, 11, 13, 15,False)
                per = np.interp(angle, (210, 310), (0, 100))
                bar = np.interp(angle, (220, 310), (650, 100))
                    # print(angle, per)

                    # Check for the dumbbell curls
                color = (255, 0, 255)
                if per == 100:
                    color = (0, 255, 0)
                    if dir == 0:
                        count += 0.5
                        dir = 1
                if per == 0:
                    color = (0, 255, 0)
                    if dir == 1:
                        count += 0.5
                        dir = 0





                display.lcd_display_string(str(int(count)), 2)



            cv2.imshow("Image", img)
            cv2.waitKey(1)


    def tricepExtention(a,b,c):
        cap = cv2.VideoCapture(0)
        detector = poseDetector()
        count = 0
        dir = 0
        while count < 10:
            success, img = cap.read()
            # img = cv2.imread("AiTrainer/test.jpg")
            img = detector.findPose(img, False)
            lmList = detector.findPosition(img, False)
            # print(lmList)
            if len(lmList) != 0:

                angle = detector.findAngle(img, a, b, c)

                    # # Left Arm
                    # angle = detector.findAngle(img, 11, 13, 15,False)
                per = np.interp(angle, (185, 255), (0, 100))
                # print(angle, per)

                # Check for the dumbbell curls
                color = (255, 0, 255)
                if per == 100:
                    color = (0, 255, 0)
                    if dir == 0:
                        count += 0.5
                        dir = 1
                if per == 0:
                    color = (0, 255, 0)
                    if dir == 1:
                        count += 0.5
                        dir = 0

                display.lcd_display_string(str(int(count)), 2)

            cv2.imshow("Image", img)
            cv2.waitKey(1)

    def sideRaise():
        cap = cv2.VideoCapture(0)
        detector = poseDetector()
        count = 0
        dir = 0
        while count < 10:
            success, img = cap.read()
            # img = cv2.imread("AiTrainer/test.jpg")
            img = detector.findPose(img, False)
            lmList = detector.findPosition(img, False)
            # print(lmList)
            if len(lmList) != 0:

                angle = detector.findAngle(img, 12, 11, 13)

                # # Left Arm
                # angle = detector.findAngle(img, 11, 13, 15,False)
                per = np.interp(angle, (175, 180), (0, 100))
                # print(angle, per)

                # Check for the dumbbell curls
                color = (255, 0, 255)
                if per == 100:
                    color = (0, 255, 0)
                    if dir == 0:
                        count += 0.5
                        dir = 1
                if per == 0:
                    color = (0, 255, 0)
                    if dir == 1:
                        count += 0.5
                        dir = 0

                display.lcd_display_string(str(int(count)), 2)

            cv2.imshow("Image", img)
            cv2.waitKey(1)
    time.sleep(2)
    display.lcd_clear()
    display.lcd_display_string("Bicep Curls", 1)
    bicepCurls()
    display.lcd_clear()
    time.sleep(2)
    
    display.lcd_display_string("Side Raise", 1)
    sideRaise()
    
    display.lcd_clear()
    time.sleep(2)
    display.lcd_display_string("Tricep Extension", 1)
    tricepExtention(12,14,16)
    
    display.lcd_clear()
    time.sleep(2)

if __name__ == "__main__":
    main()