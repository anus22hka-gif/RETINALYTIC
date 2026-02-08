import tensorflow as tf
import numpy as np
from utils.preprocess import preprocess_image

CLASS_NAMES = [
    "Normal",
    "Diabetes",
    "Glaucoma",
    "Cataract",
    "AMD",
    "Hypertension",
    "Myopia",
    "Other"
]

model = tf.keras.models.load_model("model/retina_model.h5")

def predict_retina(img_path):
    img = preprocess_image(img_path)
    preds = model.predict(img)
    class_idx = np.argmax(preds)
    confidence = float(preds[0][class_idx])
    return CLASS_NAMES[class_idx], confidence
