import json
import base64
import requests

# --- Configuration ---
# Your model name from the docker run command: -e MODEL_NAME=square_pirates_edge
MODEL_NAME = "square_pirates_edge"
# The port exposed on the host machine: -p 8501:8501
HOST_PORT = "8501"
# The path to the image you want to test
IMAGE_PATH = r"../resources/20250812_200906.jpg"
# The REST API endpoint URL
URL = f"http://localhost:{HOST_PORT}/v1/models/{MODEL_NAME}:predict"

# New URL for metadata
METADATA_URL = f"http://localhost:{HOST_PORT}/v1/models/{MODEL_NAME}/metadata"

# Get the metadata and print it
response = requests.get(METADATA_URL)
print(json.dumps(response.json(), indent=2))


# ----------------------------------------------------
# Step 1: Preprocess the Image
# ----------------------------------------------------

def prepare_image(image_path):
    """Reads an image file, encodes it in base64, and prepares the JSON payload."""

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # FIX 1: Remove .decode('utf-8') to keep the raw base64 data as bytes.
    # FIX 2: We must still pass a string/text representation to JSON,
    #        so we must encode the base64 bytes object itself.
    encoded_image_string = base64.b64encode(image_bytes).decode('utf-8')

    # FIX 3: Include both required inputs: 'image_bytes' and 'key'
    payload = {
        "instances": [
            {
                "image_bytes": {"b64": encoded_image_string},  # <--- NEW STRUCTURE
                "key": "some_unique_id_or_filename"
            }
        ]
    }
    return payload


# ----------------------------------------------------
# Step 2: Send the Request
# ----------------------------------------------------

def get_predictions(payload):
    """Sends the JSON payload to the TensorFlow Serving REST API."""

    headers = {"content-type": "application/json"}

    try:
        response = requests.post(URL, data=json.dumps(payload), headers=headers)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to TensorFlow Serving: {e}")
        return None


# ----------------------------------------------------
# Step 3: Process the Response
# ----------------------------------------------------

def extract_bounding_boxes(prediction_response):
    """Extracts bounding boxes, scores, and classes from the model response."""

    if not prediction_response or 'predictions' not in prediction_response:
        print("Invalid or empty prediction response.")
        return None

    # Object detection models typically return a dictionary of tensors,
    # which are keys within the first element of the 'predictions' list.
    model_output = prediction_response['predictions'][0]

    # Expected output keys for a Vertex AI Edge Model:
    # 'detection_boxes' (bounding boxes), 'detection_scores', 'detection_classes'

    boxes = model_output.get('detection_boxes', [])
    scores = model_output.get('detection_scores', [])
    classes = model_output.get('detection_classes', [])

    # Combine and filter results (e.g., only show results with score > 0.5)
    detections = []
    for box, score, class_id in zip(boxes, scores, classes):
        if score > 0.5:  # Confidence threshold
            detections.append({
                'box': box,  # [ymin, xmin, ymax, xmax] (normalized 0-1)
                'score': score,
                'class_id': int(class_id)  # Class ID is typically a float, cast to int
            })

    return detections


# --- Main Execution ---
if __name__ == "__main__":

    # You will need to replace this with a real image path
    # Make sure your docker container is running!

    print(f"Preparing image: {IMAGE_PATH}")
    payload = prepare_image(IMAGE_PATH)

    print(f"Sending request to {URL}...")
    response_data = get_predictions(payload)
    print(response_data)

    # if response_data:
    #     print("--- Detections Found ---")
    #     detections = extract_bounding_boxes(response_data)
    #
    #     if detections:
    #         for d in detections:
    #             print(f"Class ID: **{d['class_id']}** | Score: **{d['score']:.2f}** | Bounding Box: {d['box']}")
    #     else:
    #         print("No objects detected above the 0.5 confidence threshold.")