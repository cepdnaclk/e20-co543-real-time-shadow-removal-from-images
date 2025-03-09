import os
from flask import Flask, request, jsonify, send_from_directory, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Directories for input and output images
INPUT_DIR = "static/images/uploads"
OUTPUT_DIR = "static/images/processed"
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create the AttentionGate as a custom layer
class AttentionGate(tf.keras.layers.Layer):
    def __init__(self, inter_channel, **kwargs):
        super(AttentionGate, self).__init__(**kwargs)
        self.inter_channel = inter_channel
        self.g1_conv = tf.keras.layers.Conv2D(inter_channel, 1)
        self.x1_conv = tf.keras.layers.Conv2D(inter_channel, 1)
        self.psi_conv = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')
        
    def call(self, inputs):
        f_g, f_l = inputs  # Unpack inputs
        g1 = self.g1_conv(f_g)
        x1 = self.x1_conv(f_l)
        psi = tf.keras.activations.relu(g1 + x1)
        psi = self.psi_conv(psi)
        return tf.keras.layers.multiply([f_l, psi])
    
    def get_config(self):
        config = super(AttentionGate, self).get_config()
        config.update({"inter_channel": self.inter_channel})
        return config

# Initialize VGG for perceptual loss
vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
perceptual_model = tf.keras.models.Model(vgg.input, vgg.layers[9].output)
perceptual_model.trainable = False

# Define custom loss function
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
    
    return 0.6 * ssim_loss + 0.3 * l1_loss + 0.1 * percep_loss

# Register custom objects
custom_objects = {
    'enhanced_loss': enhanced_loss,
    'AttentionGate': AttentionGate
}

# Load the trained TensorFlow model with custom objects
model_path = "trainedModel/shadow_removal.h5"
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    print("✅ Model loaded successfully.")
else:
    raise FileNotFoundError(f"❌ Model file '{model_path}' not found!")

def preprocess_image(image):
    """Preprocess the image for the model."""
    image = image.resize((256, 256))  # Resize to match model input size
    image = np.array(image) / 127.5 - 1  # Normalize to [-1, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def postprocess_image(output):
    """Postprocess the model output to a PIL image."""
    output = (output + 1) * 127.5  # Scale back to [0, 255]
    output = np.clip(output, 0, 255).astype(np.uint8)  # Clip to valid range
    return Image.fromarray(output)

def remove_shadow(image):
    """Applies shadow removal model to an image and resizes output to match input dimensions."""
    original_size = image.size  # Store original dimensions (width, height)

    # Preprocess the image
    input_image = preprocess_image(image)

    # Generate a dummy mask (all zeros) since the model expects a mask input
    dummy_mask = np.zeros((1, 256, 256, 1), dtype=np.float32)

    # Run inference
    output = model.predict([input_image, dummy_mask])

    # Postprocess the output
    output_image = postprocess_image(output[0])

    # Resize output image to original dimensions
    output_image = output_image.resize(original_size, Image.LANCZOS)
    
    return output_image


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/remove_shadow', methods=['POST'])
def remove_shadow_api():
    """API endpoint to process an image and remove shadows."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        return jsonify({'error': 'Invalid file type'}), 400

    input_path = os.path.join(INPUT_DIR, filename)
    file.save(input_path)

    try:
        image = Image.open(input_path)
        processed_image = remove_shadow(image)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    output_path = os.path.join(OUTPUT_DIR, filename)
    processed_image.save(output_path)

    return jsonify({
        'input_url': f"/static/images/uploads/{filename}",
        'output_url': f"/static/images/processed/{filename}"
    }), 200

@app.route('/static/images/uploads/<filename>')
def get_input_image(filename):
    return send_from_directory(INPUT_DIR, filename)

@app.route('/static/images/processed/<filename>')
def get_output_image(filename):
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == '__main__':
    app.run(debug=False)  # Disable debug mode for production