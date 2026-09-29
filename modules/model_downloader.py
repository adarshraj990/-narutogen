"""
====================================================================================================
MODULE: NARUTO RVC MODEL DOWNLOADER & EXTRACTOR
====================================================================================================
Automatically fetches, verifies, and organizes the Naruto Uzumaki RVC V2 model weights and index file
from Hugging Face into project and RVC-standard directory structures.

Model Source:
https://huggingface.co/Mboisuper/Naruto_Uzumaki_315_Epochs/resolve/main/naruto-uzumaki.zip

Directories Populated:
1. models/naruto/   -> Project-standard model folder (naruto.pth, naruto.index)
2. weights/         -> Standard RVC weights folder for .pth checkpoints
3. logs/naruto/     -> Standard RVC feature index folder for .index files
====================================================================================================
"""

import os
import sys
import shutil
import zipfile
import urllib.request
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Direct download link for Naruto Uzumaki RVC V2 model (.pth + .index)
DEFAULT_NARUTO_MODEL_URL = (
    "https://huggingface.co/Mboisuper/Naruto_Uzumaki_315_Epochs/resolve/main/naruto-uzumaki.zip"
)

# Canonical filenames
PTH_CANONICAL_NAME = "naruto.pth"
INDEX_CANONICAL_NAME = "naruto.index"


def _format_bytes(bytes_count: int) -> str:
    """Format bytes to human readable string (KB, MB, GB)."""
    if bytes_count < 1024:
        return f"{bytes_count} B"
    elif bytes_count < 1024 * 1024:
        return f"{bytes_count / 1024:.1f} KB"
    elif bytes_count < 1024 * 1024 * 1024:
        return f"{bytes_count / (1024 * 1024):.2f} MB"
    else:
        return f"{bytes_count / (1024 * 1024 * 1024):.2f} GB"


def download_file_with_progress(url: str, output_path: Path, chunk_size: int = 1024 * 64) -> Path:
    """
    Downloads a remote file with a live console progress bar and download stats.
    Works with standard Python library (no external dependencies required).
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_download_path = output_path.with_suffix(output_path.suffix + ".downloading")

    print(f"\n🌐 [DOWNLOAD] Fetching archive from Hugging Face:")
    print(f"   🔗 URL: {url}")
    print(f"   📁 Destination: {output_path.name}")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    req = urllib.request.Request(url, headers=headers)

    start_time = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as response, open(temp_download_path, "wb") as out_file:
            content_length = response.headers.get("Content-Length")
            total_size = int(content_length) if content_length and content_length.isdigit() else 0
            downloaded = 0
            last_print = 0

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)

                # Update progress roughly every 0.15s or 1MB
                now = time.time()
                if now - last_print >= 0.15 or (total_size and downloaded >= total_size):
                    elapsed = max(0.001, now - start_time)
                    speed = downloaded / elapsed
                    speed_str = f"{_format_bytes(int(speed))}/s"
                    
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        bar_len = 30
                        filled = int(bar_len * downloaded / total_size)
                        bar = "█" * filled + "░" * (bar_len - filled)
                        sys.stdout.write(
                            f"\r   ⏳ [{bar}] {pct:5.1f}% | {_format_bytes(downloaded)} / {_format_bytes(total_size)} | {speed_str} "
                        )
                    else:
                        sys.stdout.write(f"\r   ⏳ Downloaded: {_format_bytes(downloaded)} | {speed_str} ")
                    sys.stdout.flush()
                    last_print = now

            print()

        # Atomic rename on completion
        if temp_download_path.exists():
            if output_path.exists():
                output_path.unlink()
            temp_download_path.rename(output_path)

        total_time = time.time() - start_time
        print(f"✅ [DOWNLOAD COMPLETE] {_format_bytes(output_path.stat().st_size)} in {total_time:.1f}s.\n")
        return output_path

    except Exception as e:
        if temp_download_path.exists():
            try:
                temp_download_path.unlink()
            except Exception:
                pass
        raise RuntimeError(f"Failed to download model weights from {url}: {e}") from e


def extract_rvc_zip(zip_path: Path, extract_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Extracts a zip file and identifies .pth and .index files inside.
    Handles archives with arbitrary internal folder nesting.
    """
    extract_dir = Path(extract_dir).resolve()
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"📦 [EXTRACT] Unpacking {zip_path.name} into {extract_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    pth_file: Optional[Path] = None
    index_file: Optional[Path] = None

    # Search recursively for .pth and .index
    for p in extract_dir.rglob("*.pth"):
        if not pth_file or p.stat().st_size > pth_file.stat().st_size:
            pth_file = p

    for p in extract_dir.rglob("*.index"):
        if not index_file or p.stat().st_size > index_file.stat().st_size:
            index_file = p

    if pth_file:
        print(f"   🎯 Found Model Weights: {pth_file.name} ({_format_bytes(pth_file.stat().st_size)})")
    else:
        print("   ⚠️ No .pth model file detected in archive!")

    if index_file:
        print(f"   🎯 Found Feature Index: {index_file.name} ({_format_bytes(index_file.stat().st_size)})")
    else:
        print("   ⚠️ No .index feature file detected in archive!")

    return pth_file, index_file


def setup_naruto_model_directories(
    discovered_pth: Path,
    discovered_index: Optional[Path],
    models_dir: Path,
    weights_dir: Path,
    logs_dir: Path,
) -> Dict[str, str]:
    """
    Distributes model and index files to both project paths and standard RVC locations:
    1. models/naruto/ (naruto.pth, naruto.index, and original names)
    2. weights/ (standard RVC weights folder)
    3. logs/naruto/ (standard RVC feature index folder)
    """
    models_dir = Path(models_dir).resolve()
    weights_dir = Path(weights_dir).resolve()
    logs_dir = Path(logs_dir).resolve()

    models_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Primary Model (.pth) placement
    primary_pth = models_dir / PTH_CANONICAL_NAME
    if discovered_pth.resolve() != primary_pth.resolve():
        shutil.copy2(discovered_pth, primary_pth)

    # Copy with original name to models_dir if different
    orig_pth_dest = models_dir / discovered_pth.name
    if orig_pth_dest.resolve() != discovered_pth.resolve() and orig_pth_dest.resolve() != primary_pth.resolve():
        shutil.copy2(discovered_pth, orig_pth_dest)

    # Copy to RVC standard weights/ folder
    rvc_weights_pth = weights_dir / discovered_pth.name
    shutil.copy2(discovered_pth, rvc_weights_pth)
    rvc_canonical_pth = weights_dir / PTH_CANONICAL_NAME
    if rvc_canonical_pth.resolve() != rvc_weights_pth.resolve():
        shutil.copy2(discovered_pth, rvc_canonical_pth)

    # 2. Index (.index) placement
    primary_index: Optional[Path] = None
    if discovered_index and discovered_index.exists():
        primary_index = models_dir / INDEX_CANONICAL_NAME
        if discovered_index.resolve() != primary_index.resolve():
            shutil.copy2(discovered_index, primary_index)

        orig_index_dest = models_dir / discovered_index.name
        if orig_index_dest.resolve() != discovered_index.resolve() and orig_index_dest.resolve() != primary_index.resolve():
            shutil.copy2(discovered_index, orig_index_dest)

        # Copy to RVC standard logs/naruto/ folder
        rvc_logs_index = logs_dir / discovered_index.name
        shutil.copy2(discovered_index, rvc_logs_index)
        rvc_canonical_index = logs_dir / INDEX_CANONICAL_NAME
        if rvc_canonical_index.resolve() != rvc_logs_index.resolve():
            shutil.copy2(discovered_index, rvc_canonical_index)

    result = {
        "model_path": str(primary_pth),
        "index_path": str(primary_index) if primary_index else "",
        "weights_path": str(rvc_weights_pth),
        "logs_path": str(logs_dir),
        "models_dir": str(models_dir),
    }

    print("\n" + "=" * 65)
    print("🍥 [NARUTO RVC SETUP COMPLETE] Directory Structure Ready:")
    print(f"   📁 Primary Weights: {result['model_path']}")
    print(f"   📑 Feature Index:   {result['index_path'] or 'None'}")
    print(f"   🗂️ RVC weights/ :   {result['weights_path']}")
    print(f"   🗂️ RVC logs/    :   {result['logs_path']}")
    print("=" * 65 + "\n")

    return result


def ensure_naruto_model(
    url: str = DEFAULT_NARUTO_MODEL_URL,
    project_root: Optional[str] = None,
    force: bool = False,
) -> Dict[str, str]:
    """
    High-level entrypoint: Checks if Naruto model already exists locally.
    If not (or if force=True), downloads the zip archive and configures directories.

    Args:
        url: HuggingFace zip download URL.
        project_root: Root directory of the repository (defaults to current working directory).
        force: If True, re-downloads even if files already exist.

    Returns:
        Dict containing model_path, index_path, weights_path, logs_path, models_dir.
    """
    root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    models_dir = root / "models" / "naruto"
    weights_dir = root / "weights"
    logs_dir = root / "logs" / "naruto"

    primary_pth = models_dir / PTH_CANONICAL_NAME
    primary_index = models_dir / INDEX_CANONICAL_NAME

    # Check for existing valid files across candidates
    pth_candidates = [
        primary_pth,
        models_dir / "naruto-uzumaki-by-mboisuper.pth",
        weights_dir / "naruto-uzumaki-by-mboisuper.pth",
        weights_dir / PTH_CANONICAL_NAME,
    ]
    index_candidates = [
        primary_index,
        models_dir / "added_IVF102_Flat_nprobe_1_naruto-uzumaki-by-mboisuper_v2.index",
        logs_dir / "added_IVF102_Flat_nprobe_1_naruto-uzumaki-by-mboisuper_v2.index",
        logs_dir / INDEX_CANONICAL_NAME,
    ]

    existing_pth = next((p for p in pth_candidates if p.exists() and p.stat().st_size > 10_000_000), None)
    existing_index = next((p for p in index_candidates if p.exists() and p.stat().st_size > 1_000_000), None)

    if not force and existing_pth:
        print(f"🍥 [NARUTO MODEL] Naruto RVC model already exists locally: {existing_pth.name}")
        # Ensure directories are in sync
        return setup_naruto_model_directories(
            discovered_pth=existing_pth,
            discovered_index=existing_index,
            models_dir=models_dir,
            weights_dir=weights_dir,
            logs_dir=logs_dir,
        )

    # Download archive
    zip_dest = root / "inputs" / "naruto-uzumaki.zip"
    zip_dest.parent.mkdir(parents=True, exist_ok=True)

    print("🍥 [NARUTO MODEL] Starting automatic download and setup...")
    downloaded_zip = download_file_with_progress(url, zip_dest)

    # Temporary extraction folder
    temp_extract = root / "inputs" / "temp_naruto_extract"
    if temp_extract.exists():
        shutil.rmtree(temp_extract, ignore_errors=True)
    temp_extract.mkdir(parents=True, exist_ok=True)

    try:
        discovered_pth, discovered_index = extract_rvc_zip(downloaded_zip, temp_extract)
        if not discovered_pth:
            raise FileNotFoundError("Could not find any .pth model weights in downloaded archive.")

        # Organize into final destinations
        model_paths = setup_naruto_model_directories(
            discovered_pth=discovered_pth,
            discovered_index=discovered_index,
            models_dir=models_dir,
            weights_dir=weights_dir,
            logs_dir=logs_dir,
        )

        # Cleanup temporary files
        try:
            shutil.rmtree(temp_extract, ignore_errors=True)
            if downloaded_zip.exists():
                downloaded_zip.unlink()
        except Exception:
            pass

        return model_paths

    except Exception as e:
        print(f"❌ [SETUP ERROR] Failed setting up Naruto model: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and setup Naruto RVC V2 model weights and index.")
    parser.add_argument("--url", type=str, default=DEFAULT_NARUTO_MODEL_URL, help="Hugging Face zip URL")
    parser.add_argument("--force", action="store_true", help="Force re-download even if already present")
    args = parser.parse_args()

    ensure_naruto_model(url=args.url, force=args.force)
