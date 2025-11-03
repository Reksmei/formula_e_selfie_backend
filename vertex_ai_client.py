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
    return {
        "data": generated_video_part.inline_data.data,
        "mime_type": generated_video_part.inline_data.mime_type
    }
