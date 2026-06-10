import cv2
import numpy as np


class ModelBoundaryBlender:
    """
    3Dモデル描画後のフレームとモデル描画前の背景フレームの差分からモデル領域を推定し、
    境界だけを背景へなじませる。

    描画済み画像に対して後処理として適用する。
    """

    def __init__(
            self,
            enabled=True,
            diff_threshold=1,
            edge_width=20,#境界の幅（ピクセル）
            blur_sigma=7,#境界のぼかしの幅
            min_area_ratio=0.001):
        self.enabled = enabled
        self.diff_threshold = diff_threshold
        self.edge_width = edge_width
        self.blur_sigma = blur_sigma
        self.min_area_ratio = min_area_ratio

    def apply(self, rendered_rgb, background_rgb):
        if not self.enabled:
            return rendered_rgb
        if rendered_rgb is None or background_rgb is None:
            return rendered_rgb
        if rendered_rgb.shape != background_rgb.shape:
            return rendered_rgb

        mask = self._estimate_model_mask(rendered_rgb, background_rgb)
        if mask is None:
            return rendered_rgb

        return self._feather_boundary(rendered_rgb, background_rgb, mask)

    def _estimate_model_mask(self, rendered_rgb, background_rgb):
        diff = cv2.absdiff(rendered_rgb, background_rgb)
        diff_gray = np.max(diff, axis=2).astype(np.uint8)
        _, mask = cv2.threshold(diff_gray, self.diff_threshold, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        mask = self._keep_large_components(mask)
        if mask is None or cv2.countNonZero(mask) == 0:
            return None

        return mask

    def _keep_large_components(self, mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if num_labels <= 1:
            return None

        h, w = mask.shape[:2]
        min_area = max(20, int(h * w * self.min_area_ratio))
        kept = np.zeros_like(mask)

        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= min_area:
                kept[labels == label] = 255

        if cv2.countNonZero(kept) == 0:
            return None

        return kept

    def _feather_boundary(self, rendered_rgb, background_rgb, mask):
        width = max(1, int(self.edge_width))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (width * 2 + 1, width * 2 + 1))

        inner_mask = cv2.erode(mask, kernel)
        outer_mask = cv2.dilate(mask, kernel)
        blend_region = outer_mask > 0

        alpha = cv2.GaussianBlur(
            mask.astype(np.float32) / 255.0,
            (0, 0),
            self.blur_sigma)
        alpha = np.maximum(alpha, inner_mask.astype(np.float32) / 255.0)
        alpha = np.clip(alpha, 0.0, 1.0)

        alpha_3ch = alpha[:, :, None]
        rendered = rendered_rgb.astype(np.float32)
        background = background_rgb.astype(np.float32)
        blended = rendered * alpha_3ch + background * (1.0 - alpha_3ch)

        output = rendered.copy()
        output[blend_region] = blended[blend_region]
        return np.clip(output, 0, 255).astype(np.uint8)
