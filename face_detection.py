from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import os
import logging
from werkzeug.utils import secure_filename

# Initialize Flask App
app = Flask(__name__)

# Initialize MTCNN detector
detector = MTCNN()

# Define folder paths
PROCESSED_FOLDER = "processed"
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)

def draw_bounding_boxes(image, faces):
    """Draws bounding boxes on detected faces and returns processed image."""
    for face in faces:
        x, y, width, height = face["box"]
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 1)

    return image

@app.route("/detect_faces", methods=["POST"])
def detect_faces():
    """API endpoint to detect faces and return image with bounding boxes."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    filename = secure_filename(file.filename)  # Sanitize filename

    try:
        # Read image directly from memory
        image_np = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Invalid image format or corrupted file.")

        # Convert to RGB for better MTCNN accuracy
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces
        faces = detector.detect_faces(image_rgb)

        if not faces:
            return jsonify({"message": "No faces detected"}), 200

        # Draw bounding boxes on original BGR image
        processed_image = draw_bounding_boxes(image, faces)

        # Save processed image
        processed_image_path = os.path.join(PROCESSED_FOLDER, f"processed_{filename}")
        cv2.imwrite(processed_image_path, processed_image)

        logging.info(f"Processed image saved: {processed_image_path}")

        return send_file(processed_image_path, mimetype="image/jpeg")

    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
