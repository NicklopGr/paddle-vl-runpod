"""
One-off warmup helper to pre-download PaddleOCR-VL models into the RunPod volume.

Run this on a GPU pod with the network volume attached, then shut it down.
Serverless containers will reuse the cached models for faster cold starts.
"""

import os
import time


def setup_model_cache() -> bool:
    """Configure model cache directory for faster cold starts."""
    network_volume_path = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume")
    model_cache_dir = os.environ.get(
        "PADDLE_VL_CACHE_DIR",
        os.path.join(network_volume_path, "paddle_models"),
    )

    if os.path.exists(network_volume_path) and os.access(network_volume_path, os.W_OK):
        os.makedirs(model_cache_dir, exist_ok=True)
        os.environ["PADDLE_HOME"] = model_cache_dir
        os.environ["PADDLEOCR_HOME"] = model_cache_dir
        os.environ["HF_HOME"] = os.path.join(model_cache_dir, "huggingface")
        os.environ["HF_HUB_CACHE"] = os.path.join(model_cache_dir, "huggingface", "hub")
        print(f"[Warmup] Using network volume cache: {model_cache_dir}")
        return True

    print("[Warmup] No network volume found, using container storage")
    return False


def main() -> None:
    setup_model_cache()
    start = time.time()
    from paddleocr import PaddleOCRVL

    _ = PaddleOCRVL()
    elapsed = time.time() - start
    print(f"[Warmup] Models ready in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
