import tensorflow as tf
import numpy as np
import cv2

def generate_gradcam(model, img_array, layer_name=None):
    """
    Generate Grad-CAM heatmap for a given image and model
    """

    # -------- Find last Conv2D layer automatically --------
    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break

    if layer_name is None:
        raise ValueError("No convolutional layer found for Grad-CAM.")

    # -------- Create gradient model --------
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(layer_name).output,
            model.output
        ]
    )

    # -------- Compute gradients --------
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    # -------- Global average pooling --------
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    # -------- Compute heatmap --------
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (224, 224))

    return heatmap
