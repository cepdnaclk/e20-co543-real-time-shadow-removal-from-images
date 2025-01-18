import os
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from opencv.grayscale import convert_to_grayscale

app = Flask(__name__, static_folder='static')

# Configure file upload settings
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['PROCESSED_FOLDER'] = os.path.join('static', 'processed')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if image and allowed_file(image.filename):
        filename = secure_filename(image.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)

        # Save the uploaded image
        image.save(input_path)

        # Process the image to grayscale
        try:
            convert_to_grayscale(input_path, output_path)
        except ValueError as e:
            return jsonify({"error": str(e)}), 500

        return jsonify({
            "uploaded_image": url_for('static', filename=f'uploads/{filename}'),
            "processed_image": url_for('static', filename=f'processed/{filename}')
        }), 200
    else:
        return jsonify({"error": "Invalid file format. Only .jpg, .jpeg, or .png allowed."}), 400

if __name__ == "__main__":
    app.run(debug=True)
