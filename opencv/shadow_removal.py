import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, losses

# Load the pre-trained VGG model and perceptual model
vgg = VGG16(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
perceptual_model = models.Model(vgg.input, vgg.layers[9].output)
perceptual_model.trainable = False

# Loss function
def enhanced_loss(y_true, y_pred):
    ssim_loss = 1 - tf.image.ssim(y_true, y_pred, max_val=2.0)
    l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    
    y_true_vgg = (y_true + 1) * 127.5  # Scale to VGG input range
    y_pred_vgg = (y_pred + 1) * 127.5
    percep_loss = tf.reduce_mean(tf.square(
        perceptual_model(y_true_vgg) - perceptual_model(y_pred_vgg)
    ))
    
    return 0.6 * ssim_loss + 0.3 * l1_loss + 0.1 * percep_loss

# Load the trained model
model = tf.keras.models.load_model("opencv/shadow_removal_model.keras", custom_objects={"enhanced_loss": enhanced_loss})

# Preprocess the image
def preprocess_image(image_path, target_size=(256, 256)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or invalid format.")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image_resized = cv2.resize(image_rgb, target_size)  # Resize to model input size
    image_normalized = image_resized / 127.5 - 1  # Normalize to [-1, 1]
    
    return image_rgb, image_normalized  # Return both original and normalized images

# Post-process the image
def postprocess_image(output_tensor, input_image):
    output_image = (output_tensor[0] + 1) * 127.5  # Convert from [-1, 1] to [0, 255]
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    
    # Compute brightness of the input image
    input_brightness = np.mean(input_image)

    # Compute brightness of the output image
    output_brightness = np.mean(output_image)

    # Adjust output image brightness if needed
    brightness_factor = input_brightness / (output_brightness + 1e-7)  # Avoid division by zero
    output_image = np.clip(output_image * brightness_factor, 0, 255).astype(np.uint8)

    return cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)  # Convert to BGR

# Function to remove shadows
def remove_shadows(input_path, output_path):
    original_image, input_image = preprocess_image(input_path)

    # Add batch dimension to the input image
    input_image = np.expand_dims(input_image, axis=0)  # Shape: (1, 256, 256, 3)

    # Create dummy masks with the same batch dimension
    dummy_masks = np.zeros((input_image.shape[0], 256, 256, 1))  # Shape: (1, 256, 256, 1)

    # Run model prediction
    output_tensor = model.predict([input_image, dummy_masks])

    # Post-process the result
    output_image = postprocess_image(output_tensor, original_image)

    # Resize output back to original size
    original_height, original_width = original_image.shape[:2]
    output_image_resized = cv2.resize(output_image, (original_width, original_height))

    # Save the final image
    cv2.imwrite(output_path, output_image_resized)

