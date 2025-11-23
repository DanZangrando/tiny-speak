# Refactoring Project Directory Structure

I have successfully reorganized the project's data structure as requested. Here is a summary of the changes:

## 1. Dictionaries
- **New Location:** `data/diccionarios/`
- **Format:** Plain text files (`.txt`).
- **Changes:**
    - Created `data/diccionarios/` directory.
    - Extracted dictionaries from `diccionarios.py` into individual text files (e.g., `tiny_kalulu_original.txt`, `basico_español.txt`).
    - Updated `diccionarios.py` to dynamically load words from these files.

## 2. Audio Datasets
- **New Location:** `data/audios/`
- **Changes:**
    - Created `data/audios/` directory.
    - Extracted `tiny-kalulu-200.tar.xz` to `data/audios/tiny_kalulu_original`.
    - Extracted `tiny-phones-200.tar.xz` to `data/audios/tiny_phones_original`.
    - Updated `pages/01_🎵_Audio_Dataset.py` to save generated audio files to `data/audios/{dataset_name}/{word}/` instead of embedding them as Base64 in the configuration.

## 3. Visual Datasets
- **New Location:** `data/visual/`
- **Changes:**
    - Created `data/visual/` directory.
    - Extracted `tiny-emnist-26.tar.xz` to `data/visual/tiny_emnist_26`.
    - Moved contents of the old `visual_dataset/` directory to `data/visual/`.
    - Removed the old `visual_dataset/` directory.
    - Updated `pages/02_🖼️_Visual_Dataset.py` to save generated images to `data/visual/`.
    - Updated `training/visual_dataset.py` to look for images in `data/visual/`.
    - Patched `master_dataset_config.json` to update file paths for existing generated images.

## 4. Cleanup
- **Deleted:** `tiny-emnist-26.tar.xz`, `tiny-kalulu-200.tar.xz`, `tiny-phones-200.tar.xz`.
- **Deleted:** Duplicate directories in `data/` (`tiny-emnist-26`, etc.) that were temporarily created during extraction.

## Verification
- The application should now function correctly using the new directory structure.
- Existing generated images should still be visible in the Visual Analytics page.
- New audio and visual generation will use the new paths.

## Next Steps
- You may want to regenerate audio datasets to fully migrate from Base64 to file-based storage, although the current implementation supports both (legacy Base64 support is included).
