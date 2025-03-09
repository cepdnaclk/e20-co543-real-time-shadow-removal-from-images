import tensorflow as tf
from tensorflow.keras import layers, models, losses
from tensorflow.keras.applications import VGG16
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Clear session
tf.keras.backend.clear_session()

# Set dataset paths for local (if you want to train the model,before run the model.py you should download the ISTD data set loacate it correct directory)
TRAIN_DIR = "./dataset/train"
TEST_DIR = "./dataset/test/test_A"


# Set dataset paths for Kaggle

# TRAIN_DIR = "/kaggle/input/istd-dataset/ISTD_Dataset/train"
# TEST_DIR = "/kaggle/input/istd-dataset/ISTD_Dataset/test/test_A"

# Define image loading function first
def load_image(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    return (img / 127.5) - 1  # Normalize to [-1, 1]

# Dataset preparation with masks
def prepare_dataset_with_masks():
    input_images = []
    target_images = []
    mask_images = []

    input_dir = os.path.join(TRAIN_DIR, "train_A")
    target_dir = os.path.join(TRAIN_DIR, "train_C")
    mask_dir = os.path.join(TRAIN_DIR, "train_B")

    # Get sorted file paths
    input_paths = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.png')])
    target_paths = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith('.png')])
    mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')])

    # Verify equal number of files
    assert len(input_paths) == len(target_paths) == len(mask_paths), "Mismatched number of files in dataset directories"

    # Load and process images
    for input_path, target_path, mask_path in zip(input_paths, target_paths, mask_paths):
        input_images.append(load_image(input_path))
        target_images.append(load_image(target_path))
        
        # Process mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (256, 256))
        mask = (mask > 127).astype(np.float32)  # Binarize mask
        mask_images.append(mask[..., np.newaxis])  # Add channel dimension

    return (
        np.array(input_images),
        np.array(mask_images),
        np.array(target_images)
    )

# Create the AttentionGate as a custom layer
class AttentionGate(tf.keras.layers.Layer):
    def __init__(self, inter_channel, **kwargs):
        super(AttentionGate, self).__init__(**kwargs)
        self.inter_channel = inter_channel
        self.g1_conv = layers.Conv2D(inter_channel, 1)
        self.x1_conv = layers.Conv2D(inter_channel, 1)
        self.psi_conv = layers.Conv2D(1, 1, activation='sigmoid')
        
    def call(self, inputs):
        f_g, f_l = inputs  # Unpack inputs
        g1 = self.g1_conv(f_g)
        x1 = self.x1_conv(f_l)
        psi = layers.Activation('relu')(layers.add([g1, x1]))
        psi = self.psi_conv(psi)
        return layers.multiply([f_l, psi])
    
    def get_config(self):
        config = super(AttentionGate, self).get_config()
        config.update({"inter_channel": self.inter_channel})
        return config
        
# Initialize VGG for perceptual loss
vgg = VGG16(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
perceptual_model = models.Model(vgg.input, vgg.layers[9].output)
perceptual_model.trainable = False

# Enhanced loss function (mask removed from loss calculation)
def enhanced_loss(y_true, y_pred):
    # Structural similarity
    ssim_loss = 1 - tf.image.ssim(y_true, y_pred, max_val=2.0)
    
    # L1 loss
    l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    
    # Perceptual loss
    y_true_vgg = (y_true + 1) * 127.5  # Scale to VGG input range
    y_pred_vgg = (y_pred + 1) * 127.5
    percep_loss = tf.reduce_mean(tf.square(
        perceptual_model(y_true_vgg) - perceptual_model(y_pred_vgg)
    ))
    
    return 0.6*ssim_loss + 0.3*l1_loss + 0.1*percep_loss

# Improved U-Net with attention gates
def build_enhanced_unet():
    inputs = layers.Input(shape=(256, 256, 3))
    masks = layers.Input(shape=(256, 256, 1))
    
    # Encoder
    e1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    e1 = layers.Conv2D(64, 3, activation='relu', padding='same')(e1)
    p1 = layers.MaxPooling2D()(e1)
    
    e2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    e2 = layers.Conv2D(128, 3, activation='relu', padding='same')(e2)
    p2 = layers.MaxPooling2D()(e2)
    
    e3 = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    e3 = layers.Conv2D(256, 3, activation='relu', padding='same')(e3)
    p3 = layers.MaxPooling2D()(e3)
    
    # Bridge
    b = layers.Conv2D(512, 3, activation='relu', padding='same')(p3)
    b = layers.Conv2D(512, 3, activation='relu', padding='same')(b)
    
    # Decoder with attention
    d3 = layers.Conv2DTranspose(256, 3, strides=2, padding='same')(b)
    # Using the custom attention gate layer
    a3 = AttentionGate(256)([d3, e3])
    d3 = layers.concatenate([a3, d3])
    d3 = layers.Conv2D(256, 3, activation='relu', padding='same')(d3)
    
    d2 = layers.Conv2DTranspose(128, 3, strides=2, padding='same')(d3)
    a2 = AttentionGate(128)([d2, e2])
    d2 = layers.concatenate([a2, d2])
    d2 = layers.Conv2D(128, 3, activation='relu', padding='same')(d2)
    
    d1 = layers.Conv2DTranspose(64, 3, strides=2, padding='same')(d2)
    a1 = AttentionGate(64)([d1, e1])
    d1 = layers.concatenate([a1, d1])
    d1 = layers.Conv2D(64, 3, activation='relu', padding='same')(d1)
    
    # Mask integration
    combined = layers.concatenate([d1, masks])
    
    outputs = layers.Conv2D(3, 1, activation='tanh')(combined)
    
    model = models.Model(inputs=[inputs, masks], outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
                  loss=enhanced_loss)
    return model

# Data augmentation using TensorFlow ops
def augment_data(input_img, mask, target_img):
    # Random left-right flip
    if tf.random.uniform(()) > 0.5:
        input_img = tf.image.flip_left_right(input_img)
        mask = tf.image.flip_left_right(mask)
        target_img = tf.image.flip_left_right(target_img)
    
    # Random up-down flip
    if tf.random.uniform(()) > 0.5:
        input_img = tf.image.flip_up_down(input_img)
        mask = tf.image.flip_up_down(mask)
        target_img = tf.image.flip_up_down(target_img)
    
    return (input_img, mask), target_img  # Return proper format

# Main execution for training
def train_model():
    # Prepare dataset
    input_images, mask_images, target_images = prepare_dataset_with_masks()

    # Create dataset pipeline
    dataset = tf.data.Dataset.from_tensor_slices((input_images, mask_images, target_images))
    dataset = dataset.shuffle(1000)
    dataset = dataset.map(lambda x, y, z: augment_data(x, y, z))
    dataset = dataset.batch(8).prefetch(tf.data.AUTOTUNE)

    # Split dataset for validation
    train_size = int(0.9 * len(input_images))
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)

    # Build and train model
    model = build_enhanced_unet()
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )

    # Save model
    model.save("shadow_removal.h5")
    print("Model saved successfully!")
    
    return model, history

if __name__ == "__main__":
    model, history = train_model()