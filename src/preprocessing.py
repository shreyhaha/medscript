"""
preprocessing.py - cleans a photo of handwriting before the model sees it.

THIS IS YOUR ECE PART. Every step is a classic image/signal-processing idea:

  1. Grayscale        colour carries no information about WHICH word it is,
                      so 3 channels -> 1 channel (less noise, less data).
  2. Median filter    (OPTIONAL, OFF by default - see below) removes
                      "salt-and-pepper" noise.
  3. CLAHE            Contrast Limited Adaptive Histogram Equalization.
                      Fixes faint ink and uneven lighting by stretching the
                      contrast LOCALLY (tile by tile), not globally.
  4. Otsu threshold   the histogram of a handwriting image has two peaks
                      (dark ink, light paper). Otsu automatically picks the
                      threshold that best separates the two peaks.
                      We use it to whiten the background, but we first
                      DILATE (grow) the ink mask by 1 pixel so the soft,
                      anti-aliased edges of thin strokes are not cut off.
  5. Crop to ink      remove empty margins so the word fills the frame.
  6. Resize + pad     every image must be the same size for the model.
                      We keep the aspect ratio (no squashing letters) and
                      pad the rest with white.

WHAT WE LEARNED BY MEASURING (validation accuracy, HOG + Logistic Regression):
    only resize, no cleaning ........................ ~68%
    full textbook pipeline (median + hard Otsu) ..... ~72%
    no median + dilated Otsu mask (FINAL) ........... ~80%
  The median filter sounded right, but these images are small and the pen
  strokes are only 1-3 pixels wide, so a 3x3 median filter erased parts of
  thin strokes. The dataset is already clean scans, so there is little
  noise to remove. On noisy phone photos you could switch it back on.
  Lesson: measure every step, don't assume.
"""
import cv2
import numpy as np

USE_MEDIAN_FILTER = False   # see the note above


def load_gray(path_or_bytes):
    """Read an image from a file path OR raw bytes (used by the web app)."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        arr = np.frombuffer(path_or_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    else:
        img = cv2.imread(str(path_or_bytes), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not read the image.")
    return to_gray(img)


def to_gray(img):
    """Step 1: any image (colour, with transparency, or grey) -> 8-bit grey."""
    if img.dtype != np.uint8:  # e.g. 16-bit PNGs
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:  # PNG with transparency: paste onto white paper
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = img[:, :, :3].astype(np.float32)
        img = (rgb * alpha + 255 * (1 - alpha)).astype(np.uint8)
    return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)


def denoise(gray):
    """Step 2 (optional): 3x3 median filter (edge-preserving noise removal)."""
    return cv2.medianBlur(gray, 3)


def enhance_contrast(gray):
    """Step 3: CLAHE - local contrast enhancement."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    return clahe.apply(gray)


def remove_background(gray, dilate=True):
    """Step 4: Otsu threshold -> make the paper pure white, keep ink grey.

    Returns (cleaned_image, ink_mask). ink_mask is True where there is ink
    (the strict Otsu mask, before dilation - used for cropping).
    """
    otsu_value, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink_mask = gray < otsu_value
    keep = ink_mask
    if dilate:
        # grow the mask by 1 pixel in every direction (3x3 kernel)
        kernel = np.ones((3, 3), np.uint8)
        keep = cv2.dilate(ink_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    cleaned = gray.copy()
    cleaned[~keep] = 255
    return cleaned, ink_mask


def crop_to_ink(gray, ink_mask, margin=4):
    """Step 5: cut away empty borders around the word."""
    # Ignore tiny specks: only count rows/cols with a few ink pixels.
    rows = np.where(ink_mask.sum(axis=1) >= 2)[0]
    cols = np.where(ink_mask.sum(axis=0) >= 2)[0]
    if len(rows) == 0 or len(cols) == 0:  # blank image: nothing to crop
        return gray
    top, bottom = max(rows[0] - margin, 0), min(rows[-1] + margin + 1, gray.shape[0])
    left, right = max(cols[0] - margin, 0), min(cols[-1] + margin + 1, gray.shape[1])
    return gray[top:bottom, left:right]


def resize_and_pad(gray, size):
    """Step 6: fit inside (height, width) keeping aspect ratio, pad with white."""
    target_h, target_w = size
    h, w = gray.shape
    scale = min(target_h / h, target_w / w)
    new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w), 255, dtype=np.uint8)
    y, x = (target_h - new_h) // 2, (target_w - new_w) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def preprocess(gray, size, clean=True, use_median=None, dilate=True):
    """Full pipeline. Input: grey image. Output: (height, width) uint8 image.

    clean=False skips steps 2-5 (only resizes). use_median / dilate let us
    switch single steps on and off to MEASURE what each one does
    (an 'ablation study' - see train_baseline.py).
    """
    if use_median is None:
        use_median = USE_MEDIAN_FILTER
    if clean:
        if use_median:
            gray = denoise(gray)
        gray = enhance_contrast(gray)
        gray, ink_mask = remove_background(gray, dilate=dilate)
        gray = crop_to_ink(gray, ink_mask)
    return resize_and_pad(gray, size)


def preprocess_steps(gray, size):
    """Return every intermediate image of the FINAL pipeline - used for the
    before/after figure and in the app, so you can SHOW each step."""
    steps = {"1. Original (grey)": gray}
    current = gray
    if USE_MEDIAN_FILTER:
        current = denoise(current)
        steps["2. Median filter"] = current
    current = enhance_contrast(current)
    steps["3. CLAHE contrast"] = current
    cleaned, ink_mask = remove_background(current)
    steps["4. Otsu background removal"] = cleaned
    steps["5. Crop to ink"] = crop_to_ink(cleaned, ink_mask)
    steps["6. Resize + pad"] = resize_and_pad(steps["5. Crop to ink"], size)
    return steps
