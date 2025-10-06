import os
import vertexai
from vertexai.generative_models import GenerativeModel, Part

# --- Configuration ---
PROJECT_ID = "formula-e-selfie"
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
IMAGE_MODEL_ID = "gemini-2.5-flash-image"
# Using a realistic, available model name for video generation.
VIDEO_MODEL_ID = "veo-2.0-generate-001"

# --- Vertex AI Initialization ---
# This block will run once when the module is imported.
try:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    image_model = GenerativeModel(IMAGE_MODEL_ID)
    video_model = GenerativeModel(VIDEO_MODEL_ID)
    print(f"Vertex AI initialized for project '{PROJECT_ID}' in location '{LOCATION}'")
except Exception as e:
    print(f"Error initializing Vertex AI: {e}")
    # You might want to handle this more gracefully depending on your application's needs
    image_model = None
    video_model = None

def generate_image(prompt_text: str, image_bytes: bytes, mime_type: str) -> dict:
    """Generates an image using Vertex AI."""
    if not image_model:
        raise ConnectionError("Vertex AI image model is not initialized.")

    print("Calling Vertex AI for image generation...")
    image_part = Part.from_data(data=image_bytes, mime_type=mime_type)
    prompt_part = Part.from_text(prompt_text)

    response = image_model.generate_content(
        [prompt_part, image_part]
    )


    print("Received image response from Vertex AI.")
    if not response.candidates:
        raise ValueError("API did not return a valid candidate for the image.")

    generated_image_part = response.candidates[0].content.parts[0]

    if not generated_image_part.inline_data.mime_type.startswith("image/"):
         raise TypeError(f"Unexpected image MIME type: {generated_image_part.inline_data.mime_type}")

    return {
        "data": generated_image_part.inline_data.data,
        "mime_type": generated_image_part.inline_data.mime_type
    }

def generate_video(image_bytes: bytes, mime_type: str) -> dict:
    """Generates a video from a single image using Vertex AI."""
    if not video_model:
        raise ConnectionError("Vertex AI video model is not initialized.")

    print("Calling Vertex AI for video generation...")
    image_part = Part.from_data(data=image_bytes, mime_type=mime_type)

    response = video_model.generate_content(
        [image_part, "make a short, 5 second video from this image"]
    )

    print("Received video response from Vertex AI.")
    if not response.candidates:
        raise ValueError("API did not return a valid candidate for the video.")

    generated_video_part = response.candidates[0].content.parts[0]

    if generated_video_part.inline_data.mime_type != "video/mp4":
        raise TypeError(f"Unexpected video MIME type: {generated_video_part.inline_data.mime_type}")

    return {
        "data": generated_video_part.inline_data.data,
        "mime_type": generated_video_part.inline_data.mime_type
    }
