import numpy as np
import tensorflow as tf

IMG_SIZE = 300

def preprocess_image_inference(image):

    # PIL → array
    img = tf.keras.utils.img_to_array(image)
    img = tf.cast(img, tf.float32)

    # ✅ Handle grayscale properly
    # Case 1: (H, W) → add channel
    if len(img.shape) == 2:
        img = tf.expand_dims(img, axis=-1)

    # Case 2: (H, W, 1) → convert to RGB
    if img.shape[-1] == 1:
        img = tf.image.grayscale_to_rgb(img)

    # Crop
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]

    crop_ratio = 0.05

    top = tf.cast(crop_ratio * tf.cast(h, tf.float32), tf.int32)
    left = tf.cast(crop_ratio * tf.cast(w, tf.float32), tf.int32)
    height = tf.cast((1 - 2 * crop_ratio) * tf.cast(h, tf.float32), tf.int32)
    width = tf.cast((1 - 2 * crop_ratio) * tf.cast(w, tf.float32), tf.int32)

    img = tf.image.crop_to_bounding_box(img, top, left, height, width)

    # Resize
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))

    img = (img / 127.5) - 1.0

    # Add batch
    img = tf.expand_dims(img, axis=0)

    return img