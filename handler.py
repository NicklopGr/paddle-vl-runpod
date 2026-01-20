"""
PaddleOCR-VL RunPod Serverless Handler

This handler processes bank statement images using PaddleOCR-VL's document parsing pipeline,
which produces structured markdown output with HTML tables.

Pipeline:
1. PP-DocLayoutV2 (Layout Analysis) - Detects 25 element categories with reading order
2. PaddleOCR-VL-0.9B (Vision-Language Recognition) - Recognizes text, tables, formulas
3. Post-processing - Outputs structured markdown with HTML tables

Input:
{
    "input": {
        "images_base64": ["base64_encoded_image_1", "base64_encoded_image_2", ...],
        // OR for single image:
        "image_base64": "base64_encoded_image"
    }
}

Output:
{
    "status": "success",
    "result": {
        "pages": [
            {
                "page_number": 1,
                "markdown": "# Document Title\n\n<table>...</table>\n\nParagraph text...",
                "json": {...}  // Optional structured JSON output
            }
        ],
        "ocrProvider": "paddleocr-vl",
        "processingTime": 1234
    }
}
"""

import runpod
import base64
import tempfile
import os
import time
from PIL import Image
import io


# Global pipeline - loaded once at container startup
paddle_vl_pipeline = None

# Network volume path for model caching (RunPod mounts at /runpod-volume)
NETWORK_VOLUME_PATH = "/runpod-volume"
MODEL_CACHE_DIR = os.path.join(NETWORK_VOLUME_PATH, "paddle_models")


def setup_model_cache():
    """Configure model cache directory for faster cold starts"""
    # Check if network volume is mounted
    if os.path.exists(NETWORK_VOLUME_PATH) and os.access(NETWORK_VOLUME_PATH, os.W_OK):
        # Create cache directory if it doesn't exist
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

        # Set PaddlePaddle and PaddleOCR cache directories
        os.environ["PADDLE_HOME"] = MODEL_CACHE_DIR
        os.environ["PADDLEOCR_HOME"] = MODEL_CACHE_DIR
        os.environ["HF_HOME"] = os.path.join(MODEL_CACHE_DIR, "huggingface")
        os.environ["HF_HUB_CACHE"] = os.path.join(MODEL_CACHE_DIR, "huggingface", "hub")

        print(f"[PaddleOCR-VL] Using network volume cache: {MODEL_CACHE_DIR}")

        # Log cached contents for debugging
        try:
            cached_items = os.listdir(MODEL_CACHE_DIR)
            if cached_items:
                print(f"[PaddleOCR-VL] Cached items: {cached_items}")
            else:
                print("[PaddleOCR-VL] Cache is empty - first run will download models")
        except Exception as e:
            print(f"[PaddleOCR-VL] Could not list cache: {e}")

        return True
    else:
        print("[PaddleOCR-VL] No network volume found, using container storage")
        return False


def load_pipeline():
    """Load PaddleOCR-VL pipeline (runs once at container startup)"""
    global paddle_vl_pipeline

    if paddle_vl_pipeline is not None:
        return paddle_vl_pipeline

    # Setup model cache before loading
    setup_model_cache()

    print("[PaddleOCR-VL] Loading pipeline...")
    start = time.time()

    from paddleocr import PaddleOCRVL

    # Initialize document parsing pipeline
    # This uses PP-DocLayoutV2 + PaddleOCR-VL-0.9B
    paddle_vl_pipeline = PaddleOCRVL()

    elapsed = time.time() - start
    print(f"[PaddleOCR-VL] Pipeline loaded in {elapsed:.2f}s")

    return paddle_vl_pipeline


def resize_image_if_needed(image: Image.Image, max_dimension: int = 1920) -> Image.Image:
    """
    Resize image if it exceeds max dimension while preserving aspect ratio.
    PaddleOCR-VL works best with images around 1080p-1920p.
    """
    width, height = image.size

    if width <= max_dimension and height <= max_dimension:
        return image

    # Calculate new dimensions preserving aspect ratio
    if width > height:
        new_width = max_dimension
        new_height = int(height * (max_dimension / width))
    else:
        new_height = max_dimension
        new_width = int(width * (max_dimension / height))

    print(f"[PaddleOCR-VL] Resizing image from {width}x{height} to {new_width}x{new_height}")
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def process_single_image(pipeline, image_base64: str, page_number: int) -> dict:
    """Process a single image and return markdown + json output"""

    # Decode base64 image
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB if necessary (PaddleOCR expects RGB)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize for optimal accuracy
    image = resize_image_if_needed(image, max_dimension=1920)

    # Save to temp file (PaddleOCR-VL requires file path)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        image.save(tmp.name, 'PNG')
        tmp_path = tmp.name

    try:
        # Run PaddleOCR-VL document parsing
        result = pipeline.predict(tmp_path)

        # Extract markdown and structured output
        # The result object has .markdown property containing the structured output
        markdown_output = getattr(result, 'markdown', None)
        json_output = getattr(result, 'json', None)

        # If markdown is a list, join pages
        if isinstance(markdown_output, list):
            markdown_output = '\n\n'.join(markdown_output)

        return {
            "page_number": page_number,
            "markdown": markdown_output or "",
            "json": json_output
        }

    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def handler(event):
    """
    RunPod serverless handler function

    Accepts:
    - images_base64: Array of base64 encoded images (multi-page)
    - image_base64: Single base64 encoded image (single page)
    """
    start_time = time.time()

    try:
        # Get input
        job_input = event.get("input", {})

        # Support both single image and multiple images
        images_base64 = job_input.get("images_base64", [])
        if not images_base64 and job_input.get("image_base64"):
            images_base64 = [job_input.get("image_base64")]

        if not images_base64:
            return {
                "status": "error",
                "error": "No images provided. Send 'images_base64' array or 'image_base64' string."
            }

        print(f"[PaddleOCR-VL] Processing {len(images_base64)} page(s)")

        # Load pipeline (cached after first call)
        pipeline = load_pipeline()

        # Process each page
        pages = []
        for i, img_b64 in enumerate(images_base64):
            page_start = time.time()
            page_result = process_single_image(pipeline, img_b64, page_number=i + 1)
            page_time = time.time() - page_start
            print(f"[PaddleOCR-VL] Page {i + 1} processed in {page_time:.2f}s")
            pages.append(page_result)

        processing_time = int((time.time() - start_time) * 1000)

        return {
            "status": "success",
            "result": {
                "pages": pages,
                "ocrProvider": "paddleocr-vl",
                "processingTime": processing_time
            }
        }

    except Exception as e:
        import traceback
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        print(f"[PaddleOCR-VL] Error: {error_msg}\n{stack_trace}")

        return {
            "status": "error",
            "error": error_msg,
            "stack_trace": stack_trace
        }


# RunPod serverless start
runpod.serverless.start({"handler": handler})
