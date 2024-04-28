import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1, detectionCon=0.8)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    img = frame.copy()
    hand, _ = detector.findHands(img)
    h, w, c = frame.shape
    if hand:
        box = hand[0]['bbox']
        x1, y1, x2, y2 = box[0] - 40, box[1] - 40, box[0] + box[2] + 40, box[1] + box[3] + 40
        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        print(x1, y1, x2, y2)
        p = frame[y1:y2, x1:x2]
        cv2.imshow('test', p)
    if cv2.waitKey(30) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
