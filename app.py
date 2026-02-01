import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os, json

BASE = "/content/drive/MyDrive/Dataset_Split"

TRAIN_DIR = os.path.join(BASE,"train")
VAL_DIR   = os.path.join(BASE,"val")

IMG_SIZE = (224,224)
BATCH = 32
EPOCH = 30

datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="categorical",
    shuffle=True)

val_gen = datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="categorical",
    shuffle=False)

NUM_CLASSES = train_gen.num_classes

base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

for layer in base_model.layers:
    layer.trainable = False

inp = Input(shape=(224,224,3))
x = base_model(inp, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
out = Dense(NUM_CLASSES, activation="softmax")(x)

model_baseline = Model(inp,out)

model_baseline.compile(
    optimizer=Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model_baseline.summary()

history_baseline = model_baseline.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCH
)

model_baseline.save("/content/drive/MyDrive/model_baseline.h5")

with open("/content/drive/MyDrive/history_baseline.json","w") as f:
    json.dump(history_baseline.history,f)

import json

with open("/content/drive/MyDrive/class_indices.json","w") as f:
    json.dump(train_gen.class_indices,f)
