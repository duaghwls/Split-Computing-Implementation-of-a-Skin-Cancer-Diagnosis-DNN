import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import models, layers, regularizers
from keras.applications import InceptionResNetV2
from keras.callbacks import ModelCheckpoint
import splitter

# # auto sharding policly?
# options = tf.data.Options()
# options.experimental_distribute.auto_shard_policy = (
#     tf.data.experimental.AutoShardPolicy.OFF
# )

# GPU 사용 설정
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.set_visible_devices(gpus[0], "GPU")

# 데이터 경로
train_dir = r"C:\project\ISIC_new\train"
val_dir = r"C:\project\ISIC_new\validation"

# 이미지 크기와 배치 크기 설정
img_width, img_height = 150, 150
batch_size = 64

# 데이터 증강(Augmentation) 설정
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

# 데이터 로딩 및 증강(Augmentation)
train_ds = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
)

val_ds = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
)


# GPU를 사용하여 모델을 병렬 처리하는 MirroredStrategy 객체 생성
strategy = tf.distribute.MirroredStrategy()


with strategy.scope():
    # 모델 생성
    # model = Sequential()
    # model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(layers.Dropout(0.2))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(layers.Dropout(0.2))
    # model.add(Conv2D(128, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Flatten())
    # model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(Dense(2, activation='softmax'))
    conv_base = InceptionResNetV2(
        weights="imagenet", include_top=False, input_shape=(150, 150, 3)
    )
    head, tail = splitter.split_model(conv_base, "block8_1")

    model = Sequential()
    model.add(head)
    model.add(tail)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.3))
    model.add(
        layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.005))
    )
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(2, activation="sigmoid"))

    # 모델 컴파일
    model.compile(
        optimizer=Adam(learning_rate=0.02),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # 모델 훈련
    history = model.fit(
        train_ds,
        steps_per_epoch=train_ds.samples // train_ds.batch_size,
        epochs=20,
        validation_data=val_ds,
        validation_steps=val_ds.samples // val_ds.batch_size,
        callbacks=[
            ModelCheckpoint(
                "SC_best_model_InceptionResNetV2_gray.h5",
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
            )
        ],
    )
