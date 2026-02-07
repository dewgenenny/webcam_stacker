import os
import subprocess
import cv2
import numpy as np
import shutil
import sys

# --- CONFIGURATION ---
STREAM_URL = "https://camsecure.co/HLS/sennenhcam.m3u8"
# We remove the extension here so Siril adds it correctly
WORK_DIR = os.path.abspath("webcam_work")
OUTPUT_BASE_NAME = os.path.abspath("final_super_res")

# Burst Settings
FRAME_COUNT = 100
FPS_CAPTURE = 10
SUPER_RES_SCALE = 2

def check_tools():
    """Ensures FFmpeg and Siril are installed."""
    if not shutil.which("ffmpeg"):
        print("Error: 'ffmpeg' not found. Install: sudo apt install ffmpeg")
        sys.exit(1)
    if not (shutil.which("siril-cli") or shutil.which("siril")):
        print("Error: 'siril' not found. Install: sudo apt install siril")
        sys.exit(1)

def capture_burst():
    """Captures frames as fast as possible."""
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)
    os.makedirs(WORK_DIR)

    print(f"[*] Burst capturing {FRAME_COUNT} frames at {FPS_CAPTURE} fps...")

    cmd = [
        "ffmpeg", "-y", "-i", STREAM_URL,
        "-vf", f"fps={FPS_CAPTURE}",
        "-frames:v", str(FRAME_COUNT),
        "-q:v", "2",
        os.path.join(WORK_DIR, "frame_%03d.jpg")
    ]

    subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL)

    if len(os.listdir(WORK_DIR)) == 0:
        print("[!] FFmpeg failed to capture images.")
        sys.exit(1)

def process_super_resolution():
    """Filters, Aligns, and Drizzles images."""
    print("[*] Analyzing frames for sharpness...")

    frames = []
    file_list = sorted([f for f in os.listdir(WORK_DIR) if f.endswith(".jpg")])

    if not file_list:
        print("Error: No frames found.")
        sys.exit(1)

    # 1. Scoring
    for f in file_list:
        path = os.path.join(WORK_DIR, f)
        img = cv2.imread(path)
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        frames.append({'path': path, 'score': score, 'img': img, 'gray': gray})

    frames.sort(key=lambda x: x['score'], reverse=True)
    best_frames = frames[:len(frames)//2]
    print(f"[*] Selected top {len(best_frames)} sharpest frames.")

    # 2. Setup Super-Res Canvas
    ref_img = best_frames[0]['img']
    ref_gray = best_frames[0]['gray']
    h, w = ref_img.shape[:2]

    aligned_dir = os.path.join(WORK_DIR, "aligned")
    os.makedirs(aligned_dir, exist_ok=True)

    ref_big = cv2.resize(ref_img, (w * SUPER_RES_SCALE, h * SUPER_RES_SCALE), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(aligned_dir, "aligned_000.png"), ref_big)

    print(f"[*] Aligning and Drizzling to {SUPER_RES_SCALE}x resolution...")

    warp_mode = cv2.MOTION_TRANSLATION
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)

    for i, frame in enumerate(best_frames[1:]):
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp_matrix = cv2.findTransformECC(
                ref_gray, frame['gray'], warp_matrix, warp_mode, criteria
            )
            
            # DEBUG: Print first few matrices
            if i < 5:
                print(f"[*] Alignment Matrix {i+1}:\n{warp_matrix}")
            
            # The matrix maps from this canvas (1280x960) back to the source frame (640x480)
            # We must scale the coordinate lookup by 1/SUPER_RES_SCALE
            # AND the translation remains in the source frame's coordinate space.
            M_final = warp_matrix.copy()
            M_final[:, :2] /= SUPER_RES_SCALE
            
            if i < 5:
                print(f"[*] Scaled Matrix {i+1}:\n{M_final}")

            aligned_img = cv2.warpAffine(
                frame['img'], M_final,
                (w * SUPER_RES_SCALE, h * SUPER_RES_SCALE),
                flags=cv2.INTER_LINEAR
            )
            
            # DEBUG: Check if aligned_img is mostly empty
            if i < 5:
                nz = np.count_nonzero(aligned_img)
                print(f"[*] Aligned Image {i+1} has {nz} non-zero pixels (Canvas size: {w*h*SUPER_RES_SCALE**2})")

            cv2.imwrite(os.path.join(aligned_dir, f"aligned_{i+1:03d}.png"), aligned_img)
            
            if (i+1) % 10 == 0:
                print(f"[*] Aligned {i+1} frames...")
        except cv2.error:
            pass

    return aligned_dir

def run_siril_stack(aligned_dir):
    """Feeds the upscaled images into Siril for stacking."""
    print(f"[*] Stacking with Siril...")

    # Clean up old files
    for f in os.listdir(aligned_dir):
        if f.endswith(".fit") or f.endswith(".fits") or f.endswith(".seq"):
            os.remove(os.path.join(aligned_dir, f))

    script_path = os.path.join(aligned_dir, "stacking.ssf")

    # We strip the extension from OUTPUT_BASE_NAME because Siril adds .png automatically
    script_content = f"""
    requires 1.2.0
    cd {aligned_dir}
    convert aligned
    stack aligned rej 3 3 -norm=addscale -out=result_stacked
    load result_stacked
    savepng {OUTPUT_BASE_NAME}
    """

    with open(script_path, "w") as f:
        f.write(script_content)

    siril_cmd = "siril-cli" if shutil.which("siril-cli") else "siril"

    subprocess.run([siril_cmd, "-s", script_path], capture_output=True, text=True)

def post_process_image():
    """Fixes the 'Dark Image' issue by stretching the histogram."""

    # Siril adds .png, so we look for that
    raw_output = OUTPUT_BASE_NAME + ".png"

    if not os.path.exists(raw_output):
        print(f"[FAIL] Expected output {raw_output} not found.")
        sys.exit(1)

    print(f"[*] Post-processing {raw_output} (Auto-Stretch)...")

    # Read image (unchanged, likely 16-bit)
    img = cv2.imread(raw_output, -1)

    if img is None:
        print("Error reading the stacked image.")
        sys.exit(1)

    # 1. Normalization (Stretch min/max to 0-255)
    #    This pulls the darkest pixel to 0 and the brightest to 255
    norm_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    norm_img = np.uint8(norm_img)

    # 2. Gamma Correction (The "Brightening" Step)
    #    Linear data looks dark. We apply Gamma 2.2 to match human vision.
    gamma = 2.2
    look_up_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    final_img = cv2.LUT(norm_img, look_up_table)

    # Save the final "Human Readable" image
    final_path = OUTPUT_BASE_NAME + "_viewable.png"
    cv2.imwrite(final_path, final_img)

    print(f"\n[SUCCESS] Final Super-Res image saved to:\n  -> {final_path}")

if __name__ == "__main__":
    check_tools()
    capture_burst()
    aligned_folder = process_super_resolution()
    run_siril_stack(aligned_folder)
    post_process_image()
