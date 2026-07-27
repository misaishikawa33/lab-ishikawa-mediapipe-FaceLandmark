import unittest

import numpy as np

from YoloRinkakuCorrector import YoloRinkakuCorrector


class YoloRinkakuCorrectorTest(unittest.TestCase):
    def setUp(self):
        self.corrector = YoloRinkakuCorrector(
            width=640,
            height=480,
            enabled=False)

    def test_frontal_yaw_uses_both_sides_when_enabled(self):
        self.assertEqual('both', self.corrector.get_rinkaku_mode_from_yaw(-15))
        self.assertEqual('both', self.corrector.get_rinkaku_mode_from_yaw(0))
        self.assertEqual('both', self.corrector.get_rinkaku_mode_from_yaw(15))

    def test_frontal_yaw_is_disabled_when_config_is_false(self):
        corrector = YoloRinkakuCorrector(
            width=640,
            height=480,
            enabled=False,
            use_front_correction=False)

        self.assertIsNone(corrector.get_rinkaku_mode_from_yaw(-14.9))
        self.assertIsNone(corrector.get_rinkaku_mode_from_yaw(0))
        self.assertIsNone(corrector.get_rinkaku_mode_from_yaw(14.9))
        self.assertEqual('right', corrector.get_rinkaku_mode_from_yaw(-15))
        self.assertEqual('left', corrector.get_rinkaku_mode_from_yaw(15))

    def test_side_yaw_enables_matching_rinkaku_correction(self):
        self.assertEqual('right', self.corrector.get_rinkaku_mode_from_yaw(-15.1))
        self.assertEqual('left', self.corrector.get_rinkaku_mode_from_yaw(15.1))

    def test_missing_yaw_disables_rinkaku_correction(self):
        self.assertIsNone(self.corrector.get_rinkaku_mode_from_yaw(None))

    def test_mask_edges_are_selected_from_upper_25_percent(self):
        contour = np.array([
            [[150, 100]],
            [[80, 120]],
            [[60, 126]],
            [[150, 200]],
            [[240, 126]],
            [[220, 120]],
        ], dtype=np.int32)

        keypoints = self.corrector.find_mask_keypoints(contour)

        self.assertEqual((80, 120), keypoints['right_edge'])
        self.assertEqual((220, 120), keypoints['left_edge'])


if __name__ == '__main__':
    unittest.main()
