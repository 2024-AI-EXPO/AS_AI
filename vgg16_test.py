from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, Flatten
from keras.initializers import Orthogonal
from keras.applications.vgg16 import VGG16
from keras.metrics import AUC, Precision, Recall
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import StreamingResponse
from cvzone.HandTrackingModule import HandDetector
import uvicorn
import cv2
import numpy as np

app = FastAPI()

size = (64, 64)
init = Orthogonal(gain=1.0, seed=None)
detector = HandDetector(maxHands=1, detectionCon=0.8)

labels_dict = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6,
    'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13,
    'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20,
    'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25, 'space': 26, 'del': 27, 'nothing': 28
}
result_dict = {labels_dict[i]: i for i in labels_dict}

on_camera = False


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
status = model.load_weights('/home/modeep3/Github/AS_AI/vgg16_model/model_weight.ckpt')
status.expect_partial()


def find(frame):
    img = frame.copy()
    hand, _ = detector.findHands(img)
    if hand:
        box = hand[0]['bbox']
        x1, y1, x2, y2 = box[0] - 70, box[1] - 70, box[0] + box[2] + 70, box[1] + box[3] + 70
        dis = ((x2 - x1) - (y2 - y1)) // 2

        if dis < 0:
            x1, x2 = x1 + dis, x2 - dis
        else:
            y1, y2 = y1 - dis, y2 + dis

        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        print(x1, y1, x2, y2)
        crop = frame[y1:y2, x1:x2]
        cv2.imshow('test', crop)
        return crop
    return 


def preprocessing(_frame):
    _frame = cv2.resize(_frame, size)
    image = np.array([_frame])
    image = image.astype('float32')/255.0
    return image


def generate_frames(camera):
    cap = cv2.VideoCapture(0)
    time = 0
    buffer = ''  # 그냥 버퍼
    sentence = ''  # 문장 만들기
    while camera:
        ret, frame = cap.read()
        if not ret:
            print('카메라 오류 발생')
            break
        hand_img = find(frame)
        frame = cv2.resize(frame, dsize=(900, 720), interpolation=cv2.INTER_LINEAR)
        frame = cv2.flip(frame, 1)
        if hand_img is not None:

            pre_image = preprocessing(hand_img)
            result = model.predict(pre_image).squeeze()
            idx = int(np.argmax(result))
            text = result_dict[idx]

            if time >= 20:
                if text == 'space':
                    sentence += '_'
                elif text == 'del':
                    sentence = sentence[:-1]
                else:
                    l = 0
                    while sentence and sentence[-1] == '_':
                        sentence = sentence[:-1]
                        l += 1
                    sentence += ' ' * l
                    sentence += text
                time = 0

            if text == buffer:
                buffer = ''
                time += 1
            else:
                buffer = text

            cv2.putText(
                frame,
                text,
                org=(200, 300),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2,
                color=(0, 0, 0),
                thickness=2,
                lineType=cv2.LINE_AA
            )

            if sentence:
                cv2.putText(
                    frame,
                    sentence,
                    org=(50, 450),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 255, 0),
                    thickness=2,
                    lineType=cv2.LINE_AA
                )
        else:
            sentence = ''
        frame_encoded = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_encoded + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()


@app.on_event("startup")
async def start_up():
    global on_camera
    on_camera = True


@app.on_event("shutdown")
async def shut_down():
    global on_camera
    on_camera = False


@app.get("/AI")
async def stream_frames(backgroundtasks: BackgroundTasks):
    backgroundtasks.add_task(generate_frames, on_camera)
    return StreamingResponse(generate_frames(on_camera), media_type="multipart/x-mixed-replace;boundary=frame")

if __name__ == '__main__':
    uvicorn.run(app="vgg16_test:app",
                host="127.0.0.1",
                port=5955,
                reload=True,
                workers=1)