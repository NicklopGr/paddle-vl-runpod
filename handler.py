"""
PaddleOCR-VL-1.5 RunPod Serverless Handler

Processes bank statement images using PaddleOCR-VL-1.5's document parsing pipeline,
producing structured markdown output with HTML tables.

Pipeline:
1. PP-DocLayoutV2 (Layout Analysis) - Detects 25 element categories with reading order
2. PaddleOCR-VL-1.5-0.9B (Vision-Language Recognition) - Recognizes text, tables, formulas
3. Post-processing - Outputs structured markdown with HTML tables

Input:
{
    "input": {
        // URL mode (preferred - avoids base64 overhead):
        "image_urls": ["https://...", "https://...", ...],
        // OR base64 mode:
        "images_base64": ["base64_encoded_image_1", ...],
        // OR single image:
        "image_base64": "base64_encoded_image",
        // Optional:
        "skip_resize": false,
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
                "markdown": "...",
                "parsing_res_list": [...],
                "json": {...}
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
import warnings
import requests
from PIL import Image
import io
import numpy as np

# ============================================================================
# PERFORMANCE OPTIMIZATIONS - Set before any PaddlePaddle imports
# ============================================================================

# Skip model verification on startup (models already cached and verified)
os.environ["PADDLEX_SKIP_MODEL_CHECK"] = "1"

# Suppress PaddlePaddle API compatibility warnings (torch.split differences)
# These are benign warnings about PyTorch vs PaddlePaddle API differences
warnings.filterwarnings("ignore", message=".*Non compatible API.*")
warnings.filterwarnings("ignore", category=Warning, module="paddle.utils.decorator_utils")

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
    # This uses PP-DocLayoutV2 + PaddleOCR-VL-1.5-0.9B
    paddle_vl_pipeline = PaddleOCRVL(pipeline_version="v1.5")

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


def download_image(url: str) -> bytes:
    """Download image from URL (e.g. Azure Blob SAS URL)"""
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return resp.content


def prepare_temp_file(image_bytes: bytes, index: int, skip_resize: bool) -> str:
    """Save image bytes to a temp file, optionally resizing. Returns temp file path."""
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    if not skip_resize:
        image = resize_image_if_needed(image, max_dimension=1920)

    tmp = tempfile.NamedTemporaryFile(suffix=f'_page{index}.png', delete=False, dir='/tmp')
    image.save(tmp.name, 'PNG')
    tmp.close()
    return tmp.name


def extract_page_result(res, page_number: int) -> dict:
    """Extract markdown and structured data from a single PaddleOCR-VL result."""
    markdown_output = ""
    parsing_res_list = []
    json_output = None

    try:
        md_info = res.markdown
        if md_info:
            if isinstance(md_info, dict):
                md_texts = md_info.get('markdown_texts', '')
                if isinstance(md_texts, str):
                    markdown_output += md_texts
                elif isinstance(md_texts, list):
                    markdown_output += '\n\n'.join(str(t) for t in md_texts)
            elif isinstance(md_info, str):
                markdown_output += md_info
    except Exception as e:
        print(f"[PaddleOCR-VL] Error accessing markdown (page {page_number}): {e}")

    try:
        json_data = res.json
        if json_data:
            json_output = convert_to_serializable(json_data)
            if isinstance(json_data, dict) and 'parsing_res_list' in json_data:
                parsing_res_list = convert_to_serializable(json_data['parsing_res_list'])
    except Exception as e:
        print(f"[PaddleOCR-VL] Error accessing json (page {page_number}): {e}")

    # Fallback: build markdown from parsing_res_list
    if not markdown_output and parsing_res_list:
        for block in parsing_res_list:
            label = block.get('block_label', '')
            content = block.get('block_content', '')
            if content:
                if label == 'table':
                    markdown_output += f"\n\n{content}\n\n"
                else:
                    markdown_output += f"\n{content}\n"

    return {
        "page_number": page_number,
        "markdown": markdown_output.strip(),
        "parsing_res_list": parsing_res_list,
        "json": json_output
    }


def handler(event):
    """
    RunPod serverless handler function

    Accepts (in priority order):
    - image_urls: Array of URLs (SAS URLs from Azure Blob)
    - images_base64: Array of base64 encoded images
    - image_base64: Single base64 encoded image
    """
    start_time = time.time()

    try:
        job_input = event.get("input", {}) or {}

        # Warmup-only path
        if event.get("warmup") or job_input.get("warmup"):
            load_pipeline()
            return {
                "status": "success",
                "result": {
                    "warmup": True,
                    "cache_dir": MODEL_CACHE_DIR
                }
            }

        # Collect image bytes from URLs or base64 (URL takes priority)
        image_urls = job_input.get("image_urls", [])
        images_base64 = job_input.get("images_base64", [])
        if not images_base64 and job_input.get("image_base64"):
            images_base64 = [job_input.get("image_base64")]

        skip_resize = job_input.get("skip_resize", False)
        if skip_resize:
            print(f"[PaddleOCR-VL] skip_resize=True (client handled sizing)")

        # Determine input mode
        image_bytes_list = []
        if image_urls:
            print(f"[PaddleOCR-VL] URL mode: downloading {len(image_urls)} image(s)")
            for i, url in enumerate(image_urls):
                dl_start = time.time()
                image_bytes_list.append(download_image(url))
                print(f"[PaddleOCR-VL] Downloaded page {i+1} in {time.time()-dl_start:.2f}s ({len(image_bytes_list[-1])} bytes)")
        elif images_base64:
            print(f"[PaddleOCR-VL] Base64 mode: {len(images_base64)} image(s)")
            for img_b64 in images_base64:
                image_bytes_list.append(base64.b64decode(img_b64))
        else:
            return {
                "status": "error",
                "error": "No images provided. Send 'image_urls', 'images_base64', or 'image_base64'."
            }

        print(f"[PaddleOCR-VL] Processing {len(image_bytes_list)} page(s)")

        # Load pipeline (cached after first call)
        pipeline = load_pipeline()

        # Save all images to temp files
        temp_paths = []
        try:
            for i, img_bytes in enumerate(image_bytes_list):
                tmp_path = prepare_temp_file(img_bytes, i + 1, skip_resize)
                temp_paths.append(tmp_path)
                print(f"[PaddleOCR-VL] Prepared temp file for page {i+1}: {tmp_path}")

            # Batch predict: pass all paths at once for SDK-level batching
            predict_start = time.time()
            results = list(pipeline.predict(temp_paths))
            predict_time = time.time() - predict_start
            print(f"[PaddleOCR-VL] Batch predict completed in {predict_time:.2f}s for {len(temp_paths)} page(s)")

            # Map results to pages
            pages = []
            for i, res in enumerate(results):
                page_result = extract_page_result(res, page_number=i + 1)
                print(f"[PaddleOCR-VL] Page {i+1} markdown length: {len(page_result['markdown'])}")
                pages.append(page_result)

        finally:
            # Cleanup all temp files
            for tmp_path in temp_paths:
                try:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                except Exception:
                    pass

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
