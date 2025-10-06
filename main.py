import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from google.cloud import storage

# Import the new client functions
from vertex_ai_client import generate_image, generate_video

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)

# --- Cloud Storage Initialization ---
# The service should be run with the following service account:
# gemini-deployment-sa@formula-e-selfie.iam.gserviceaccount.com
storage_client = storage.Client()
# TODO: Replace with your bucket names
CREATED_IMAGES_BUCKET = "created-images"
CREATED_VIDEOS_BUCKET = "created-videos"

# --- API Endpoints ---
@app.route("/")
def health_check():
    return "OK", 200

@app.route("/generate", methods=["POST"])
def handle_generate_image():
    """Handles the initial image generation request."""
    print("Received request on /generate")
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    prompt = request.form.get('prompt')
    if not prompt:
        return jsonify({"error": "No prompt provided."}), 400

    image_file = request.files['image']
    filename = secure_filename(image_file.filename)
    print(f"Processing image: {filename} with prompt: '{prompt[:30]}...'")

    try:
        image_bytes = image_file.read()
        mime_type = image_file.mimetype
        if filename.endswith(".webp"):
            mime_type = "image/webp"

        # Call the refactored function from the client
        result = generate_image(prompt, image_bytes, mime_type)

        # Upload the image to Cloud Storage
        bucket = storage_client.bucket(CREATED_IMAGES_BUCKET)
        blob = bucket.blob(f"{filename}")
        blob.upload_from_string(result["data"], content_type=result["mime_type"])

        # Generate a signed URL for the image
        signed_url = blob.generate_signed_url(version="v4", expiration=3600)

        # Return the signed URL of the image
        return jsonify({"imageData": signed_url})

    except Exception as e:
        print(f"An error occurred during image generation: {e}")
        if "safety" in str(e).lower():
            return jsonify({"error": "Image generation failed due to safety filters."}), 500
        else:
            return jsonify({"error": f"An internal server error occurred: {e}"}), 500

@app.route("/generate-video", methods=["POST"])
def handle_generate_video():
    """Handles the video generation request using an uploaded image."""
    print("Received request on /generate-video")
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided for video generation."}), 400

    image_file = request.files['image']
    filename = secure_filename(image_file.filename)
    print(f"Processing image for video: {filename}")

    try:
        image_bytes = image_file.read()
        mime_type = image_file.mimetype

        # Call the refactored function from the client
        result = generate_video(image_bytes, mime_type)

        # Upload the video to Cloud Storage
        bucket = storage_client.bucket(CREATED_VIDEOS_BUCKET)
        blob = bucket.blob(f"{filename.split('.')[0]}.mp4")
        blob.upload_from_string(result["data"], content_type=result["mime_type"])

        # Generate a signed URL for the video
        signed_url = blob.generate_signed_url(version="v4", expiration=3600)

        # Return the signed URL of the video
        return jsonify({"videoData": signed_url})

    except Exception as e:
        print(f"An error occurred during video generation: {e}")
        if "safety" in str(e).lower():
            return jsonify({"error": "Video generation failed due to safety filters."}), 500
        else:
            return jsonify({"error": f"An internal server error occurred: {e}"}), 500

if __name__ == '__main__':
    # IMPORTANT: This is for local development only.
    # Use a production-ready WSGI server like Gunicorn or uWSGI for deployment.
    app.run(debug=True, host='0.0.0.0', port=8080)
