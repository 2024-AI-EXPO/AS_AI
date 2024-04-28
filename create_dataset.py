# 손만 나오게 만들기

import cv2
import os

alphabet = 'B'
# ls = [i for i in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
# ls.extend(['space', 'del']

data_dir = f"dataset/{alphabet}"
cap = cv2.VideoCapture(0)
data_num = 2000

os.makedirs(data_dir, exist_ok=True)

for i in range(data_num):
    ret, frame = cap.read()
    if not ret:
        break
    flip = cv2.flip(frame, 1)
    resizing = cv2.resize(flip, (224, 224))

    cv2.imshow('frame', flip)
    cv2.imwrite(data_dir + f'/{alphabet}_{i+1}.jpg', resizing)

    if cv2.waitKey(30) == ord('q'):
        break
