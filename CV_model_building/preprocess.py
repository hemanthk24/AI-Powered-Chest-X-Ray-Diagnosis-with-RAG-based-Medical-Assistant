import numpy as np
import tensorflow as tf

import tensorflow as tf

def preprocess_image_inference(image):

    # Ensure tensor
    image = tf.convert_to_tensor(image)

    # Ensure float32 FIRST
    image = tf.cast(image, tf.float32)

    # 🔥 Handle grayscale cases
    if len(image.shape) == 2:
        # (H, W) → (H, W, 1)
        image = tf.expand_dims(image, axis=-1)

    if image.shape[-1] == 1:
        # (H, W, 1) → (H, W, 3)
        image = tf.image.grayscale_to_rgb(image)

    # Crop (same as training)
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]

    crop_ratio = 0.05

    crop_top = tf.cast(crop_ratio * tf.cast(h, tf.float32), tf.int32)
    crop_left = tf.cast(crop_ratio * tf.cast(w, tf.float32), tf.int32)

    crop_height = tf.cast((1 - 2 * crop_ratio) * tf.cast(h, tf.float32), tf.int32)
    crop_width = tf.cast((1 - 2 * crop_ratio) * tf.cast(w, tf.float32), tf.int32)

    image = tf.image.crop_to_bounding_box(
        image,
        crop_top,
        crop_left,
        crop_height,
        crop_width
    )

    # Resize (same)
    image = tf.image.resize(image, (300, 300))

    # Add batch dimension
    image = tf.expand_dims(image, axis=0)

    return image