from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, Flatten
from keras.initializers import Orthogonal
from keras.applications.vgg16 import VGG16
from keras.metrics import AUC
from keras.metrics import Precision
from keras.metrics import Recall

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
size = (64, 64)
init = Orthogonal(gain=1.0, seed=None)

labels_dict = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6,
    'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13,
    'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20,
    'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25, 'space': 26, 'del': 27, 'nothing': 28
}
result_dict = {labels_dict[i]: i for i in labels_dict}


def create_model():
    vgg = Sequential([
        Input(shape=(size[0], size[1], 3)),
        VGG16(weights='imagenet', include_top=False),
        Flatten(),
        Dense(420, activation='relu'),
        Dropout(0.4),
        Dense(29, activation='softmax')
    ])

    vgg.compile(
        optimizer='sgd',
        loss='categorical_crossentropy',
        metrics=[
            AUC(name='Accuracy'),
            Precision(name='Precision'),
            Recall(name='Recall')
        ]
    )
    return vgg


model = create_model()

status = model.load_weights('vgg16_model/model_weight.ckpt')
status.expect_partial()


def preprocessing(_frame):
    _frame = cv2.resize(_frame, size)
    image = np.array([_frame])
    image = image.astype('float32')/255.0
    return image


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('카메라 오류 발생')
        break

    pre_image = preprocessing(frame)
    result = model.predict(pre_image).squeeze()
    idx = int(np.argmax(result))
    text = result_dict[idx]

    frame = cv2.resize(frame, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)
    frame = cv2.flip(frame, 1)

    cv2.putText(
        frame,
        text,
        org=(540, 320),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=3,
        color=(255, 255, 255),
        thickness=2,
        lineType=cv2.LINE_AA
    )
    cv2.imshow('ASL_Transform', frame)

    if cv2.waitKey(10) == ord('q'):
        print('프로그램을 종료합니다.')
        break

cap.release()
cv2.destroyAllWindows()
