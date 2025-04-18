import base64
import os
import time
from pathlib import Path
from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
#MODEL_NAME = os.getenv("CLAUDE_MODEL_NAME", "claude-3-5-haiku-20241022")
#MODEL_NAME = os.getenv("CLAUDE_MODEL_NAME", "claude-3-haiku-20240307")

if not api_key:
    raise ValueError("ANTHROPIC_API_KEY is not set in environment variables.")

# Initialize Anthropic client
client = Anthropic(api_key=api_key)

def get_base64_encoded_image(image_path):
    """Encodes an image file in base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def create_batch_requests(image_folder):
    """Creates batch requests for processing images."""
    requests = []
    for idx, image_file in enumerate(Path(image_folder).glob("*.png")):  # Adjust extension as needed
        requests.append(
            Request(
                custom_id=f"bubble_{idx}",
                params=MessageCreateParamsNonStreaming(
                    model=MODEL_NAME,
                    max_tokens=1024,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": get_base64_encoded_image(str(image_file))
                                }
                            },
                            {
                                "type": "text",
                                "text": "Extract and transcribe the text from this image. Only output the text, nothing else."
                            }
                        ]
                    }]
                )
            )
        )
    return requests

def get_batch_results(batch_id):
    """Polls for batch completion and retrieves results."""
    while True:
        batch_status = client.messages.batches.get(batch_id)
        if batch_status.status in ["completed", "failed"]:
            break
        print("Waiting for batch to complete...")
        time.sleep(5)  # Adjust polling interval as needed
    
    results = {}
    for result in client.messages.batches.results(batch_id):
        match result.result.type:
            case "succeeded":
                for content_item in result.result.message.content:
                    if content_item["type"] == "text":
                        results[result.custom_id] = content_item["text"]
                print(f"Success: {result.custom_id}")
            case "errored":
                print(f"Error processing {result.custom_id}: {result.result.error}")
            case "expired":
                print(f"Request expired: {result.custom_id}")
    
    return results

def process_manga_bubbles(image_folder):
    """Submits images for text extraction and processes the results."""
    requests = create_batch_requests(image_folder)
    message_batch = client.messages.batches.create(requests=requests)
    print(f"Batch {message_batch.id} submitted with {len(requests)} requests")
    return get_batch_results(message_batch.id)

def process_single_image(image_path):
    """Sends a single image request and returns the response."""
    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": get_base64_encoded_image(image_path)
                    }
                },
                {
                    "type": "text",
                    "text": "Extract and transcribe the text from this image. Only output the text, nothing else."
                }
            ]
        }]
    )
    
    # Extract text content from response
    for content_block in message.content:
        if content_block.type == "text":
            return content_block.text
    
    return "No text found in response."

if __name__ == "__main__":
    image_folder = "path/to/your/bubble/images"
    # results = process_manga_bubbles(image_folder)
    
    # # Save batch results to file
    # with open("transcriptions.txt", "w", encoding="utf-8") as f:
    #     for bubble_id, text in sorted(results.items()):
    #         f.write(f"{bubble_id}: {text}\n")
    
    single_image_path = os.path.abspath(os.getenv("INPUT_BUBBLE", "./data/bubbles/test.jpg"))
    single_result = process_single_image(single_image_path)
    print("Test Image Transcription:", single_result)
