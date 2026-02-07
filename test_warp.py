import cv2
import numpy as np
import os

def test_warp_logic():
    # Create a 100x100 dummy image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(img, (50, 50), 20, (255, 255, 255), -1) # White circle in center
    cv2.putText(img, "TOP", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    scale = 2
    w_big, h_big = 100 * scale, 100 * scale
    
    # 1. Stretched (what we want)
    # M maps from big_coords (0..200) to small_coords (0..100)
    # small_x = 0.5 * big_x
    M_stretched = np.array([[0.5, 0, 0], [0, 0.5, 0]], dtype=np.float32)
    stretched = cv2.warpAffine(img, M_stretched, (w_big, h_big))
    
    # 2. Quadrant (current bug)
    # M = Identity
    # small_x = big_x
    M_quadrant = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    quadrant = cv2.warpAffine(img, M_quadrant, (w_big, h_big))
    
    # 3. Tiny (what user sees now?)
    # M = 4.0?
    # small_x = 4.0 * big_x
    M_tiny = np.array([[4.0, 0, 0], [0, 4.0, 0]], dtype=np.float32)
    tiny = cv2.warpAffine(img, M_tiny, (w_big, h_big))
    
    cv2.imwrite("test_stretched.png", stretched)
    cv2.imwrite("test_quadrant.png", quadrant)
    cv2.imwrite("test_tiny.png", tiny)
    print("Test images generated.")

if __name__ == "__main__":
    test_warp_logic()
