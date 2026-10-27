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

    def test_mask_edges_are_selected_from_upper_15_percent(self):
        contour = np.array([
            [[150, 100]],
            [[80, 110]],
            [[60, 126]],
            [[150, 200]],
            [[240, 126]],
            [[220, 110]],
        ], dtype=np.int32)

        keypoints = self.corrector.find_mask_keypoints(contour)

        self.assertEqual((80, 110), keypoints['right_edge'])
        self.assertEqual((220, 110), keypoints['left_edge'])

    def test_outlier_filter_uses_yolo_crop_border(self):
        crop_rect = (100, 50, 500, 400)
        keypoints = {
            'chin': (300, 350),
            'right_edge': (116, 120),
            'left_edge': (480, 120),
        }

        for mode in ('right', 'left', 'both'):
            self.assertFalse(
                self.corrector.is_yolo_keypoints_reliable(
                    keypoints,
                    mode,
                    crop_rect))

        keypoints['right_edge'] = (118, 120)
        keypoints['left_edge'] = (482, 120)
        for mode in ('right', 'left', 'both'):
            self.assertTrue(
                self.corrector.is_yolo_keypoints_reliable(
                    keypoints,
                    mode,
                    crop_rect))

    def test_chin_is_rejected_only_at_crop_border(self):
        crop_rect = (100, 50, 500, 400)
        keypoints = {
            'chin': (300, 399),
            'right_edge': (150, 120),
            'left_edge': (450, 120),
        }

        self.assertFalse(
            self.corrector.is_yolo_keypoints_reliable(
                keypoints,
                'both',
                crop_rect))

        # 5%安全領域の外でも、枠線に載っていなければ顎は採用する。
        keypoints['chin'] = (300, 398)
        self.assertTrue(
            self.corrector.is_yolo_keypoints_reliable(
                keypoints,
                'both',
                crop_rect))

    def test_reset_clears_stale_landmark_overrides(self):
        self.corrector.landmark_overrides_px = {152: (100, 200)}
        self.corrector.landmark_overrides_loaded = True

        self.corrector.reset_latest_detection(keep_crop=True)

        self.assertEqual({}, self.corrector.landmark_overrides_px)
        self.assertFalse(self.corrector.landmark_overrides_loaded)


    def test_yolo_rotation_and_translation_can_be_selected_independently(self):
        yolo_rotation = np.full((3, 3), 1.0)
        base_rotation = np.full((3, 3), 2.0)
        yolo_translation = np.full(3, 3.0)
        base_translation = np.full(3, 4.0)

        cases = [
            (True, True, yolo_rotation, yolo_translation),
            (True, False, yolo_rotation, base_translation),
            (False, True, base_rotation, yolo_translation),
            (False, False, base_rotation, base_translation),
        ]
        for use_rotation, use_translation, expected_R, expected_t in cases:
            with self.subTest(
                    use_rotation=use_rotation,
                    use_translation=use_translation):
                self.corrector.use_yolo_rotation = use_rotation
                self.corrector.use_yolo_translation = use_translation
                R, t = self.corrector.combine_pose_components(
                    yolo_rotation,
                    yolo_translation,
                    base_rotation,
                    base_translation)
                np.testing.assert_array_equal(expected_R, R)
                np.testing.assert_array_equal(expected_t, t)


if __name__ == '__main__':
    unittest.main()
