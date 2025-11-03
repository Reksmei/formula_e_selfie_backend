import traceback
import base64
import io
import uuid
import requests
import google.auth
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from google.cloud import storage
import qrcode

# Import the new client functions
from vertex_ai_client import generate_image, generate_video

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)

# --- Cloud Storage Initialization ---
storage_client = storage.Client()
CREATED_IMAGES_BUCKET = "created-images"
CREATED_VIDEOS_BUCKET = "created-videos"

# --- API Endpoints ---
@app.route("/")
def health_check():
    return "OK", 200

@app.route("/generate", methods=["POST"])
def handle_generate_image():
    """Handles the initial image generation request."""
    try:
        print("Received request on /generate")
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided."}), 400

        prompt = request.form.get('prompt')
        if not prompt:
            return jsonify({"error": "No prompt provided."}), 400

        image_file = request.files['image']
        filename = f"{uuid.uuid4()}-{secure_filename(image_file.filename)}"
        print(f"Processing image: {filename} with prompt: '{prompt[:30]}...'")

        image_bytes = image_file.read()
        mime_type = image_file.mimetype
        if filename.endswith(".webp"):
            mime_type = "image/webp"

        print("Calling generate_image...")
        result = generate_image(prompt, image_bytes, mime_type)
        print("generate_image successful.")

        print("Uploading to Cloud Storage...")
        bucket = storage_client.bucket(CREATED_IMAGES_BUCKET)
        blob = bucket.blob(f"{filename}")
        blob.upload_from_string(result["data"], content_type=result["mime_type"])
        print("Upload to Cloud Storage successful.")

        print("Generating public URL...")
        public_url = f"https://storage.googleapis.com/{CREATED_IMAGES_BUCKET}/{filename}"
        print("Public URL generated successfully.")

        print("Generating QR code...")
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        print(f"QR code data: {public_url}")
        qr.add_data(public_url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        
        buf = io.BytesIO()
        img.save(buf)
        buf.seek(0)
        
        qr_code_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        print("QR code generated successfully.")

        return jsonify({"imageData": public_url, "qrCode": qr_code_base64})

    except Exception as e:
        print(f"An error occurred during image generation: {e}")
        traceback.print_exc()
        if "safety" in str(e).lower():
            return jsonify({"error": "Image generation failed due to safety filters."}), 500
        else:
            return jsonify({"error": f"An internal server error occurred: {e}"}), 500

@app.route("/generate-video", methods=["POST"])
def handle_generate_video():
    """Handles the video generation request using an uploaded image."""
    try:
        print("Received request on /generate-video")
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided for video generation."}), 400

        prompt = request.form.get('prompt')
        if not prompt:
            return jsonify({"error": "No prompt provided."}), 400

        image_file = request.files['image']
        image_bytes = image_file.read()
        mime_type = image_file.mimetype

        print("Calling generate_video...")
        operation_name = generate_video(image_bytes, mime_type, prompt)
        print(f"generate_video started with operation: {operation_name}")

        return jsonify({"operationName": operation_name})

    except Exception as e:
        print(f"An error occurred during video generation: {e}")
        traceback.print_exc()
        return jsonify({"error": f"An internal server error occurred: {e}"}), 500

@app.route("/video-status/<path:operation_name>", methods=["GET"])
def handle_video_status(operation_name):
    """Checks the status of a long-running video generation operation."""
    try:
        print(f"Received request on /video-status for operation: {operation_name}")
        credentials, project_id = google.auth.default()
        access_token = credentials.token

        endpoint = f"https://us-central1-aiplatform.googleapis.com/v1/{operation_name}"
        headers = {"Authorization": f"Bearer {access_token}"}

        response = requests.get(endpoint, headers=headers)
        response.raise_for_status()
        operation = response.json()

        if operation.get("done", False):
            print("Video generation operation is complete.")
            # Extract video data from the operation response
            # This part might need adjustment based on the actual response structure
            video_data_b64 = operation["response"]["predictions"][0]["bytesBase64Encoded"]
            video_data = base64.b64decode(video_data_b64)
            video_filename = f"{uuid.uuid4()}.mp4"

            # Upload to GCS
            bucket = storage_client.bucket(CREATED_VIDEOS_BUCKET)
            blob = bucket.blob(video_filename)
            blob.upload_from_string(video_data, content_type="video/mp4")
            public_url = f"https://storage.googleapis.com/{CREATED_VIDEOS_BUCKET}/{video_filename}"

            # Generate QR code
            qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=4)
            qr.add_data(public_url)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")
            buf = io.BytesIO()
            img.save(buf)
            buf.seek(0)
            qr_code_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            return jsonify({"status": "done", "videoData": public_url, "qrCode": qr_code_base64})
        else:
            print("Video generation operation is still in progress.")
            return jsonify({"status": "processing"})

    except Exception as e:
        print(f"An error occurred while checking video status: {e}")
        traceback.print_exc()
        return jsonify({"error": f"An internal server error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
