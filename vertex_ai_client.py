import os
import base64
import google.auth
from google.auth.transport.requests import AuthorizedSession
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig, SafetySetting, HarmCategory

# --- Configuration ---
PROJECT_ID = "formula-e-selfie"
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
IMAGE_MODEL_ID = "gemini-2.5-flash-image"
VIDEO_MODEL_ID = "veo-3.1-generate-preview"

# --- Vertex AI Initialization ---
try:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    image_model = GenerativeModel(IMAGE_MODEL_ID)
    print(f"Vertex AI initialized for project '{PROJECT_ID}' in location '{LOCATION}'")
except Exception as e:
    print(f"Error initializing Vertex AI: {e}")
    image_model = None

def generate_image(prompt_text: str, image_bytes: bytes, mime_type: str) -> dict:
    """Generates an image using Vertex AI."""
    if not image_model:
        raise ConnectionError("Vertex AI image model is not initialized.")

    print("Calling Vertex AI for image generation...")
    image_part = Part.from_data(data=image_bytes, mime_type=mime_type)
    prompt_part = Part.from_text(prompt_text)

    response = image_model.generate_content(
        [prompt_part, image_part],
        generation_config=GenerationConfig(
            temperature=0.4,
            top_p=1.0,
            top_k=32,
            max_output_tokens=2048,
        ),
        safety_settings=[
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]
    )

    print(response)


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

def generate_video(image_bytes: bytes, mime_type: str, prompt_text: str = "make a short, 5 second video from this image") -> str:
    """Generates a video from a single image using Vertex AI by calling the long-running endpoint."""
    print("Calling Vertex AI for video generation...")

    credentials, project_id = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
    authed_session = AuthorizedSession(credentials)

    endpoint = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{VIDEO_MODEL_ID}:predictLongRunning"

    request_body = {
        "instances": [
            {
                "prompt": prompt_text,
                "image_bytes": base64.b64encode(image_bytes).decode("utf-8"),
            }
        ],
        "parameters": {
            "video_length": "5",
            "fps": "24"
        }
    }

    response = authed_session.post(endpoint, json=request_body)
    response.raise_for_status()

    operation_name = response.json()["name"]
    print(f"Started video generation operation: {operation_name}")

    return operation_name
