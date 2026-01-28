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
        "image_base64": "base64_encoded_image",
        // Optional: skip handler-side resize (client handles sizing)
        "skip_resize": false,
        // Optional: warmup-only request to pre-download models
        "warmup": true
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
                "parsing_res_list": [...],  // Full structured parsing results
                "json": {...}  // Full JSON output
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
import numpy as np


# Global pipeline - loaded once at container startup
paddle_vl_pipeline = None

# Network volume path for model caching (RunPod mounts at /runpod-volume)
NETWORK_VOLUME_PATH = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume")
MODEL_CACHE_DIR = os.environ.get(
    "PADDLE_VL_CACHE_DIR",
    os.path.join(NETWORK_VOLUME_PATH, "paddle_models"),
)


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


def convert_to_serializable(obj):
    """Convert numpy arrays and other non-serializable types to JSON-safe types"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, tuple):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj


def process_single_image(pipeline, image_base64: str, page_number: int, skip_resize: bool = False) -> dict:
    """Process a single image and return markdown + structured output

    Args:
        pipeline: The PaddleOCR-VL pipeline
        image_base64: Base64 encoded image
        page_number: Page number (1-indexed)
        skip_resize: If True, skip handler-side resize (image already sized by client)
    """

    # Decode base64 image
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB if necessary (PaddleOCR expects RGB)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize for optimal accuracy (unless client already handled it)
    if skip_resize:
        print(f"[PaddleOCR-VL] Skipping resize, using original {image.size[0]}x{image.size[1]}")
    else:
        image = resize_image_if_needed(image, max_dimension=1920)

    # Save to temp file (PaddleOCR-VL requires file path)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        image.save(tmp.name, 'PNG')
        tmp_path = tmp.name

    try:
        # Run PaddleOCR-VL document parsing
        # predict() returns an iterator/generator of result objects
        results = pipeline.predict(tmp_path)

        markdown_output = ""
        parsing_res_list = []
        json_output = None

        # Iterate through results (usually one per image)
        for res in results:
            print(f"[PaddleOCR-VL] Result object type: {type(res)}")
            print(f"[PaddleOCR-VL] Result attributes: {dir(res)}")

            # Try to get markdown content
            # PaddleOCR 3.x: res.markdown is a dict with 'markdown_texts' key
            try:
                md_info = res.markdown
                print(f"[PaddleOCR-VL] markdown type: {type(md_info)}")
                if md_info:
                    if isinstance(md_info, dict):
                        md_texts = md_info.get('markdown_texts', '')
                        print(f"[PaddleOCR-VL] markdown_texts type: {type(md_texts)}")
                        if isinstance(md_texts, str):
                            markdown_output += md_texts
                        elif isinstance(md_texts, list):
                            markdown_output += '\n\n'.join(str(t) for t in md_texts)
                    elif isinstance(md_info, str):
                        markdown_output += md_info
            except Exception as e:
                print(f"[PaddleOCR-VL] Error accessing markdown: {e}")

            # Try to get JSON/parsing_res_list
            try:
                json_data = res.json
                print(f"[PaddleOCR-VL] json type: {type(json_data)}")
                if json_data:
                    json_output = convert_to_serializable(json_data)
                    # Extract parsing_res_list from json
                    if isinstance(json_data, dict) and 'parsing_res_list' in json_data:
                        parsing_res_list = convert_to_serializable(json_data['parsing_res_list'])
                        print(f"[PaddleOCR-VL] Found {len(parsing_res_list)} blocks in parsing_res_list")
            except Exception as e:
                print(f"[PaddleOCR-VL] Error accessing json: {e}")

            # If no markdown but we have parsing_res_list, build markdown from blocks
            if not markdown_output and parsing_res_list:
                print("[PaddleOCR-VL] Building markdown from parsing_res_list")
                for block in parsing_res_list:
                    label = block.get('block_label', '')
                    content = block.get('block_content', '')
                    if content:
                        if label == 'table':
                            markdown_output += f"\n\n{content}\n\n"
                        else:
                            markdown_output += f"\n{content}\n"

        print(f"[PaddleOCR-VL] Final markdown length: {len(markdown_output)}")
        print(f"[PaddleOCR-VL] Parsing blocks: {len(parsing_res_list)}")

        return {
            "page_number": page_number,
            "markdown": markdown_output.strip(),
            "parsing_res_list": parsing_res_list,
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
        job_input = event.get("input", {}) or {}

        # Warmup-only path (pre-download models into the cache volume)
        if event.get("warmup") or job_input.get("warmup"):
            load_pipeline()
            return {
                "status": "success",
                "result": {
                    "warmup": True,
                    "cache_dir": MODEL_CACHE_DIR
                }
            }

        # Support both single image and multiple images
        images_base64 = job_input.get("images_base64", [])
        if not images_base64 and job_input.get("image_base64"):
            images_base64 = [job_input.get("image_base64")]

        if not images_base64:
            return {
                "status": "error",
                "error": "No images provided. Send 'images_base64' array or 'image_base64' string."
            }

        # Check if client wants to skip handler-side resize
        skip_resize = job_input.get("skip_resize", False)
        if skip_resize:
            print(f"[PaddleOCR-VL] skip_resize=True (client handled sizing)")

        print(f"[PaddleOCR-VL] Processing {len(images_base64)} page(s)")

        # Load pipeline (cached after first call)
        pipeline = load_pipeline()

        # Process each page
        pages = []
        for i, img_b64 in enumerate(images_base64):
            page_start = time.time()
            page_result = process_single_image(pipeline, img_b64, page_number=i + 1, skip_resize=skip_resize)
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
