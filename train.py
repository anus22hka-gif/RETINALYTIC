import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# =============================
# CONFIG
# =============================
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 50   # 🔥 50 phases
NUM_CLASSES = 8  # ODIR-5K has 8 disease classes

DATASET_DIR = "ODIR-5K"
TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "Training Images")
LABEL_FILE = os.path.join(DATASET_DIR, "data.xlsx")

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# =============================
# LOAD LABELS
# =============================
df = pd.read_excel(LABEL_FILE)

# Keep required columns
df = df[["Left-Fundus", "Right-Fundus",
         "N", "D", "G", "C", "A", "H", "M", "O"]]

# Convert multi-label → single label (argmax)
df["label"] = df[["N","D","G","C","A","H","M","O"]].values.argmax(axis=1)

# Use LEFT eye images only (standard practice)
df["filename"] = df["Left-Fundus"]
df = df[["filename", "label"]].dropna()

df["label"] = df["label"].astype(str)

# =============================
# DATA GENERATORS
# =============================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

train_gen = datagen.flow_from_dataframe(
    dataframe=df,
    directory=TRAIN_IMG_DIR,
    x_col="filename",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_dataframe(
    dataframe=df,
    directory=TRAIN_IMG_DIR,
    x_col="filename",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# =============================
# MODEL
# =============================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False  # Phase 1: feature extraction

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =============================
# CALLBACKS
# =============================
callbacks = [
    ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "best_retina_model.h5"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=4,
        verbose=1
    )
]

# =============================
# TRAINING (50 PHASES)
# =============================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

# =============================
# SAVE FINAL MODEL
# =============================
model.save(os.path.join(MODEL_DIR, "retina_odir_final.h5"))

print("✅ Training complete")
print("📁 Best model saved as: model/best_retina_model.h5")
print("📁 Final model saved as: model/retina_odir_final.h5")
