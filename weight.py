from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, Flatten
from keras.initializers import Orthogonal
from keras.applications.vgg16 import VGG16
from keras.metrics import AUC
from keras.metrics import Precision
from keras.metrics import Recall

size = (64, 64)
init = Orthogonal(gain=1.0, seed=None)


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
