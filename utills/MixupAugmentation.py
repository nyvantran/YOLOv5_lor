import albumentations as A
from albumentations.core.transforms_interface import DualTransform
import numpy as np
import cv2
from typing import Dict, Any, List, Tuple


class MixupAugmentationV2(A.DualTransform):
    def __init__(self, mixup_alpha, p=1.0):
        self.mixup_alpha = mixup_alpha
        self.p = p
        super().__init__(p=p)

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        mixup_image = data["mixup_image"]
        mixup_bboxes = data["mixup_bboxes"]

        return {
            'lam': np.random.beta(self.mixup_alpha, self.mixup_alpha),
            'mixup_image': mixup_image,
            'mixup_bboxes': mixup_bboxes,
        }

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        mixup_image = params["mixup_image"]
        lam = params["lam"]
        mixuped_img = lam * img + (1 - lam) * mixup_image
        return mixuped_img.astype(np.uint8)

    def apply_to_bboxes(self, bboxes, **params):
        mixup_bboxes = params["mixup_bboxes"]
        mixuped_bboxes = np.concatenate((bboxes, mixup_bboxes), axis=0)
        return mixuped_bboxes
