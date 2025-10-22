import numpy as np
from PIL import Image
import cv2
from .config import CROP_TOP, CROP_BOTTOM, CROP_LEFT, CROP_RIGHT

def redact_burned_text(pil_img: Image.Image) -> Image.Image:
    gray = np.array(pil_img.convert("L"))
    h, w = gray.shape[:2]
    y0 = int(h*CROP_TOP); y1 = int(h*(1-CROP_BOTTOM))
    x0 = int(w*CROP_LEFT); x1 = int(w*(1-CROP_RIGHT))
    crop = gray[y0:y1, x0:x1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,3))
    blackhat = cv2.morphologyEx(crop, cv2.MORPH_BLACKHAT, kernel)
    blackhat = cv2.GaussianBlur(blackhat, (3,3), 0)
    _, th = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
    th = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_RECT,(5,3)), 1)

    cleaned = cv2.inpaint(crop, th, 3, cv2.INPAINT_TELEA)
    return Image.fromarray(cleaned)
