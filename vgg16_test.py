from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, Flatten
from keras.initializers import Orthogonal
from keras.applications.vgg16 import VGG16
from keras.metrics import AUC, Precision, Recall
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import StreamingResponse
import uvicorn
import cv2
import numpy as np

app = FastAPI()

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
status = model.load_weights('/Users/yabbi/Desktop/GitHub/AS_AI/vgg16_model/model_weight.ckpt')
status.expect_partial()


def preprocessing(_frame):
    _frame = cv2.resize(_frame, size)
    image = np.array([_frame])
    image = image.astype('float32')/255.0
    return image


def generate_frames(camera):
    time = 0
    buffer = ''  # 그냥 버퍼
    sentence = ''  # 문장 만들기
    while camera:
        ret, frame = cap.read()
        if not ret:
            print('카메라 오류 발생')
            break

        pre_image = preprocessing(frame)
        result = model.predict(pre_image).squeeze()
        idx = int(np.argmax(result))
        text = result_dict[idx]

        frame = cv2.resize(frame, dsize=(900, 720), interpolation=cv2.INTER_LINEAR)
        frame = cv2.flip(frame, 1)

        if time >= 20:
            if text == 'space':
                sentence += '_'
            elif text == 'del':
                if sentence and sentence[-1] == '_':
                    sentence = sentence[:-1]
                sentence = sentence[:-1]
            elif text != 'nothing':
                if sentence and sentence[-1] == '_':
                    sentence = sentence[:-1]
                    sentence += ' '
                sentence += text
            # else:
            #     sentence = ''
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
            color=(255, 255, 255),
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
                color=(255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA
            )
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
    on_camera =False
    
    

@app.get("/AI")
async def stream_frames(backgroundtasks: BackgroundTasks):
    backgroundtasks.add_task(generate_frames, on_camera)
    return StreamingResponse(generate_frames(on_camera), media_type="multipart/x-mixed-replace;boundary=frame")

if __name__ == '__main__':
    uvicorn.run(app ="vgg16_test:app",
                host="0.0.0.0",
                port=5955,
                reload=False,
                workers=1)