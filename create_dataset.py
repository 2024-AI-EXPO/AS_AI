import cv2
import os

train_dir = "dataset/train/A"
test_dir = "dataset/test/A"
# ls = [i for i in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
# ls.extend(['space', 'del'])
cap = cv2.VideoCapture(0)
data_num = 50

os.makedirs(train_dir, exist_ok=True)

for i in range(data_num):
    ret, frame = cap.read()
    if not ret:
        break
    resizing = cv2.resize(frame, (224, 224))
    flip = cv2.flip(frame, 1)

    cv2.imshow('frame', flip)
    cv2.imwrite(train_dir + f'/A_{i+1}.jpg', frame)

    if cv2.waitKey(30) == ord('q'):
        break
