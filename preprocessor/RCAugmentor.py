import cv2
import numpy as np
import random

class RCAugmentor:
    """
    훈련 전용 증강 
    """
    def __init__(self, hflip_prob=0.5, brightness_delta=0.2, blur_prob=0.3):
        self.hflip_prob = hflip_prob
        self.brightness_delta = brightness_delta
        self.blur_prob = blur_prob
        self.flip_map = {
        10: 170,   # 왼쪽 끝 <-> 오른쪽 끝
        40: 140,
        70: 110,
        90: 90,    # 직진 <-> 직진
        110: 70,
        140: 40,
        170: 10}

    def __call__(self, img_bgr: np.ndarray, angle: int):
        # 좌우 플립
        if random.random() < self.hflip_prob:
            img_bgr = cv2.flip(img_bgr, 1)
            angle = self.flip_map.get(angle, angle)

        # 밝기 변화
        if self.brightness_delta > 0:
            alpha = 1.0 + random.uniform(-self.brightness_delta, self.brightness_delta)
            img_bgr = np.clip(img_bgr.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

        # 블러
        if random.random() < self.blur_prob:
            img_bgr = cv2.GaussianBlur(img_bgr, (3, 3), 0)

        return img_bgr, angle
