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
    if hand:
        box = hand[0]['bbox']
        cv2.rectangle(frame, (box[0] - 40, box[1] - 40),
                      (box[0] + box[2] + 40, box[1] + box[3] + 40),
                      (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('hand', frame)
    if cv2.waitKey(30) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
