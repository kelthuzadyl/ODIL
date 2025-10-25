import rclpy
import torch
import numpy as np
import matplotlib.pyplot as plt

from match_servoers.base_servoer import VisualServoer
from lightglue import LightGlue, SIFT, SuperPoint, viz2d
from lightglue.utils import numpy_image_to_torch, rbd


class LightGlueVisualServoer(VisualServoer):
    """
    Visual servoing using LightGlue with either SuperPoint or SIFT feature extractors.
    """

    def __init__(
        self,
        rgb_ref: np.ndarray,
        seg_ref: np.ndarray,
        features: str = "superpoint",
        use_depth: bool = True,
        silent: bool = False,
        visualize_matches: bool = False,
    ):
        """
        Initialize the LightGlue visual servoer.

        Args:
            rgb_ref: Reference RGB image.
            seg_ref: Reference segmentation mask.
            features: 'superpoint' or 'sift'.
            use_depth: Whether to use depth data during observation.
            silent: Disable logging if True.
            visualize_matches: Whether to visualize matches for debugging.
        """
        super().__init__(use_depth=use_depth, silent=silent)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rgb_ref = rgb_ref
        self.seg_ref = seg_ref
        self.features = features.lower()
        self.visualize_matches = visualize_matches

        # Initialize extractor and matcher
        if self.features == "sift":
            self.extractor = SIFT(backend="pycolmap", max_num_keypoints=1024).eval().to(self.device)
            self.matcher = LightGlue(features="sift", depth_confidence=-1, width_confidence=-1).eval().to(self.device)
        elif self.features == "superpoint":
            self.extractor = SuperPoint(max_num_keypoints=1024).eval().to(self.device)
            self.matcher = LightGlue(
                features="superpoint",
                depth_confidence=-1,
                width_confidence=-1,
                filter_threshold=0.1,
            ).eval().to(self.device)
        else:
            raise NotImplementedError(f"Feature extractor '{features}' is not supported.")

        # Extract features from reference image
        self.feats0 = self.extractor.extract(numpy_image_to_torch(self.rgb_ref).to(self.device))

    def match_lightglue(self, filter_seg_ref: bool = True, apply_seg_ref_to_current_rgb: bool = False):
        """
        Match features between the reference image and the current RGB frame.

        Args:
            filter_seg_ref: If True, use the reference segmentation mask to filter matches.
            apply_seg_ref_to_current_rgb: If True, apply the segmentation mask to the current RGB frame. Near convergence only.

        Returns:
            Tuple of (scored ref_keypoints, scored live_keypoints, live_depth) or (None, None, None) on failure.
        """
        live_rgb, live_depth = self.observe()

        if apply_seg_ref_to_current_rgb and self.seg_ref is not None:
            live_rgb = live_rgb * self.seg_ref[:, :, None]

        if live_rgb is None:
            self.log_error("No RGB image received. Check camera and topics.")
            return None, None, None

        try:
            # Extract features from live image
            feats1 = self.extractor.extract(numpy_image_to_torch(live_rgb).to(self.device))

            # Match reference and live features
            matches01 = self.matcher({"image0": self.feats0, "image1": feats1})
            feats0, feats1, matches01 = [rbd(x) for x in [self.feats0, feats1, matches01]]

            matches = matches01["matches"]
            scores = matches01["scores"]

            if matches.shape[0] == 0:
                self.log_warn("No matches found between reference and current frame.")
                return None, None, None

            # Convert to numpy
            mkpts_0 = feats0["keypoints"][matches[..., 0]].cpu().numpy()
            mkpts_1 = feats1["keypoints"][matches[..., 1]].cpu().numpy()
            scores = scores.detach().cpu().numpy()[..., None]

            # segmentation filtering
            if filter_seg_ref:
                mkpts_0, mkpts_1, scores = self._filter_by_segmentation(mkpts_0, mkpts_1, scores)

            if mkpts_0 is None or mkpts_0.shape[0] == 0:
                self.log_warn("No valid keypoints after segmentation filtering.")
                return None, None, None

            mkpts_scores_0 = np.concatenate((mkpts_0, scores), axis=1)
            mkpts_scores_1 = np.concatenate((mkpts_1, scores), axis=1)

            self.log_info(f"Matched {len(mkpts_scores_0)} keypoints.")

            # Optional visualization
            if self.visualize_matches:
                self._visualize_matches(mkpts_0, mkpts_1, live_rgb)

            return mkpts_scores_0, mkpts_scores_1, live_depth

        except Exception as e:
            self.log_error(f"Error in match_lightglue: {e}")
            return None, None, None

    def _filter_by_segmentation(self, mkpts_0, mkpts_1, scores):
        """Filter keypoints based on the reference segmentation mask."""
        coords = mkpts_0.astype(int)
        h, w = self.seg_ref.shape[:2]

        valid = (
            (coords[:, 0] >= 0) & (coords[:, 0] < w) &
            (coords[:, 1] >= 0) & (coords[:, 1] < h)
        )
        coords = coords[valid]

        mask = np.zeros(mkpts_0.shape[0], dtype=bool)
        mask[valid] = self.seg_ref[coords[:, 1], coords[:, 0]].astype(bool)

        if mask.sum() == 0:
            return None, None, None

        return mkpts_0[mask], mkpts_1[mask], scores[mask]

    def _visualize_matches(self, mkpts_0, mkpts_1, live_rgb):
        """Visualize keypoint matches between reference and live images."""
        axes = viz2d.plot_images([self.rgb_ref, live_rgb])
        viz2d.plot_matches(mkpts_0, mkpts_1, color="lime", lw=0.2)
        viz2d.plot_keypoints([mkpts_0, mkpts_1], colors="blue")
        plt.show()

    def run(self):
        pass

def main():
    """Test LightGlue visual servoing node."""
    rclpy.init()

    # Example usage with dummy reference images
    rgb_ref = np.zeros((480, 640, 3), dtype=np.uint8)
    seg_ref = np.ones((480, 640), dtype=np.uint8)

    servoer = LightGlueVisualServoer(
        rgb_ref,
        seg_ref,
        features="superpoint",
        use_depth=True,
        silent=False,
        visualize_matches=False,
    )

    while rclpy.ok():
        servoer.match_lightglue(filter_seg_ref=True)
        rclpy.spin_once(servoer)


if __name__ == "__main__":
    main()
