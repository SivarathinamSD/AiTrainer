import cv2
import mediapipe as mp
import time
import math
import numpy as np
count= 0
dir = 0
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
        self.results = self.pose.process(imgRGB)  # Store results as an instance variable
        if draw:
            if self.results.pose_landmarks:
                self.mpDraw.draw_landmarks(
                    img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosition(self, img, draw=False):
        self.lmList = []
        if hasattr(self, 'results') and self.results.pose_landmarks:  # Check if 'results' exists
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                visibility = lm.visibility
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, visibility, cx, cy])
        return self.lmList
    def findAngle(self, img, p1,p2,p3, draw=True):
       x1, y1 = self.lmList[p1][2:]
       x2, y2 = self.lmList[p2][2:]
       x3, y3 = self.lmList[p3][2:]
       angle = math.degrees( math.atan2(y1-y2,x1-x2)-math.atan2(y3-y2,x3-x2))
       if angle < 0:
            angle =abs(angle)
       if draw:
            cv2.line(img, (x2,y2),(x1,y1),(255,0,255),3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 0, 255), 3)

            cv2.circle(img, (x1,y1), 15, (255,0,0), 2)
            cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 0), 2)
            cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (255, 0, 0), 2)
            cv2.circle(img, (x3, y3), 10, (255, 0, 0), cv2.FILLED)
            cv2.putText(img, str(int(angle)),(x2-20,y2 +50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)

            return angle
def main():
    cap = cv2.VideoCapture(0)
    detector = poseDetector()
    count = 0
    dir = 0
    while True:

        success, img = cap.read()
        img = cv2.flip(img, 2)
        lmList= detector.getPosition(img, draw=False)
        img = detector.findPose(img, draw = False)

        if lmList: #optional
            angle = detector.findAngle(img,12,14,16)
            per = np.interp(angle, (30,180),(0,100))
            if per == 100:
                if dir == 0:
                    count += 0.5
                    dir += 1
            if per == 0:
                if dir == 1:
                    count += 0.5
                    dir += 0
            cv2.putText(img, str(int(count)), (100,100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        cv2.imshow("Image", img)

        cv2.waitKey(1)

if __name__ == "__main__":
    main()
