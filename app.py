# app.py
from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import cv2
import io
import base64
import matplotlib.pyplot as plt

# Create a Flask web application instance
app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to generate pencil sketches from an uploaded image
def generate_pencil_sketch(input_img):
    # Read the input image using OpenCV
    org_img = cv2.imread(input_img)

    # Convert the image to RGB format
    original_img_rgb = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)

    # Invert the grayscale image
    inverted_gray_image = cv2.bitwise_not(gray_image)

    # Apply Gaussian blur to the inverted grayscale image
    blurred_img = cv2.GaussianBlur(inverted_gray_image, (111, 111), 0)
    
    # Invert the blurred image
    inverted_blurred_image = cv2.bitwise_not(blurred_img)

    # Create a pencil sketch by dividing the grayscale image by the inverted blurred image
    pencil_sketch = cv2.divide(gray_image, inverted_blurred_image, scale=220)

    # Create a Matplotlib figure to display images
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs = axs.flatten()

    # Define a list of tuples containing image titles and the corresponding images
    images = [
        ('Original Image', original_img_rgb),
        ('Black & White Image', gray_image),
        ('Inverted Grey Image', inverted_gray_image),
        ('Blurred Image', blurred_img),
        ('Inverted Blurred Image', inverted_blurred_image),
        ('Pencil Sketch', pencil_sketch)
    ]

    # Display images in the Matplotlib figure
    for i, (title, img) in enumerate(images):
        axs[i].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        axs[i].axis('off')
        axs[i].set_title(title)

    # Adjust layout for better visualization
    plt.tight_layout()

    # Convert the Matplotlib figure to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    base64_image = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Close the Matplotlib figure to avoid potential issues
    plt.close(fig)

    return base64_image

# Define the route for the web application
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        # Retrieve the uploaded file
        file = request.files['file']

        # If the user does not select a file, return an error message
        if file.filename == '':
            return render_template('index.html', error='No selected file')

        # If the file has an allowed extension, proceed with processing
        if file and allowed_file(file.filename):
            # Securely save the file to the specified upload folder
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Generate pencil sketches and convert to base64
            base64_image = generate_pencil_sketch(filepath)

            # Render the template with the generated sketches
            return render_template('index.html', filename=filename, base64_image=base64_image)

    # Render the default template for the GET request
    return render_template('index.html')

# Run the Flask application if this script is executed
if __name__ == '__main__':
    app.run(debug=True)
