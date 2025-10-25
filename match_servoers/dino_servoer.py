
# Instructions to set up DINO feature extractor for real-time visual servoing:
# 1. Install this DINO repo to extract correspondences: https://github.com/ShirAmir/dino-vit-features
# 2. mv dino-vit-features dino_vit_features
# 3. Open ~/ODIL-VS/dino_vit_features/correspondences.py and change from extractor import ViTExtractor to from .extractor import ViTExtractor
# 4. Replace the find_correspondence method in dino_vit_features.correspondences for real-time visual servoing. You can find a modified version at the end of this file.

import numpy as np
from shtab import DIR
import torch
from PIL import Image
import matplotlib.pyplot as plt
import rclpy
import mediapy as media
import os

from dino_vit_features.correspondences import find_correspondences, draw_correspondences
from dino_vit_features.extractor import ViTExtractor
from match_servoers.base_servoer import VisualServoer

class DINOVisualServoer(VisualServoer):
    """
    Real-time Visual servoing using DINO correspondences.
    """

    def __init__(
        self,
        DIR: str,
        rgb_ref: np.ndarray,
        seg_ref: np.ndarray,
        use_depth: bool = True,
        silent: bool = False,
        visualize_matches: bool = False,
    ):
        """
        Initialize the DINO visual servoer.

        Args:
            DIR: Directory containing the reference images and depth maps.
            rgb_ref: Reference RGB image.
            seg_ref: Reference segmentation mask.
            use_depth: Whether to use depth data during observation.
            silent: Disable logging if True.
            visualize_matches: Whether to visualize matches for debugging.
        """
        super().__init__(use_depth=use_depth, silent=silent)

        self.DIR = DIR
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rgb_ref = rgb_ref
        self.seg_ref = seg_ref
        self.visualize_matches = visualize_matches

        # Cache for the reference image to avoid repetitive extraction and speed up matching
        self.mkpts_0 = None

        # DINO feature extractor parameters
        self.num_pairs = 8 #@param
        self.load_size = 480 #@param
        self.layer = 9 #@param
        self.facet = 'key' #@param
        self.bin = True #@param
        self.thresh = 0.2 #@param
        self.model_type = 'dino_vits8' #@param
        self.stride = 8 #@param
        self.patch_size = 8 #not changeable
        
        # Initialize the DINO feature extractor
        self.extractor = ViTExtractor(self.model_type, self.stride, device=self.device)

        # Initialize descriptor cache variables
        try:
            cache = np.load(f'{self.DIR}/dino_desc.npz')
            self.desc1 = torch.tensor(cache['desc1'], device='cuda:0')
            self.descriptor_vectors = torch.tensor(cache['descriptor_vectors'], device='cuda:0')
            self.num_patches = tuple(cache['num_patches'])
        except Exception as e:
            self.log_warn(f"Failed to load DINO cache: {e}. Initializing empty cache.")
            self._initialize_and_cache_reference_features()

    def _initialize_and_cache_reference_features(self):
        """
           Initialize features for the reference image.
           This method extracts and caches DINO descriptors for the reference image.
        """

        # Extract descriptors for reference image using the extractor directly
        with torch.no_grad():
            patches_xy, self.desc1, self.descriptor_vectors, self.num_patches = self.extract_descriptors(image_1_path=f"{self.DIR}/ref_rgb_wrist.png", image_2_path=f"{self.DIR}/ref_rgb_wrist.png", num_pairs=self.num_pairs, load_size=self.load_size)
            
        # Save descriptor vectors and desc1 to cache for future use
        try:
            np.savez(f'{self.DIR}/dino_desc.npz', desc1=self.desc1.cpu().numpy(), descriptor_vectors=self.descriptor_vectors.cpu().numpy(), num_patches=self.num_patches)
        except Exception as e:
            self.log_warning(f"Failed to save DINO cache: {e}.")


    def match_dino(self, filter_seg_ref=True):
        live_rgb, live_depth = self.observe()

        if live_rgb is None:
            self.log_error("No RGB image received. Check camera and topics.")
            return None, None, None

        # Extract descriptors and original images
        descriptors_list, _ = self.extract_desc_maps([live_rgb], load_size=self.load_size)

        if self.mkpts_0 is None:
            # Get keypoints for the reference image
            key_y, key_x = self.extract_descriptor_nn(self.descriptor_vectors, emb_im=self.desc1, 
                                                patched_shape=self.num_patches, return_heatmaps=False)
            mkpts_0 = np.array([(y * live_rgb.shape[0] / self.num_patches[0], x * live_rgb.shape[1] / self.num_patches[1]) 
                                for y, x in zip(key_y, key_x)])

            self.mkpts_0 = mkpts_0
        
        # Get keypoints for the live image
        key_y, key_x = self.extract_descriptor_nn(self.descriptor_vectors, emb_im=descriptors_list[0], 
                                            patched_shape=self.num_patches, return_heatmaps=False)
        mkpts_1 = np.array([(y * live_rgb.shape[0] / self.num_patches[0], x * live_rgb.shape[1] / self.num_patches[1]) 
                            for y, x in zip(key_y, key_x)])
    

        if self.visualize_matches:
            fig = draw_correspondences(self.mkpts_0, mkpts_1, Image.fromarray(self.rgb_ref), live_rgb)
            plt.show()

        # Swap x and y
        mkpts_0, mkpts_1 = self.mkpts_0[:, ::-1], mkpts_1[:, ::-1]

        if filter_seg_ref:
            coords = mkpts_0.astype(int)
            mask = self.seg_ref[coords[:, 1], coords[:, 0]]

            mkpts_0 = mkpts_0[mask]
            mkpts_1 = mkpts_1[mask]
                    
        return mkpts_0, mkpts_1, live_depth 

    def extract_descriptors(self, image_1_path, image_2_path, num_pairs=None, load_size=None):
        num_pairs = num_pairs or self.num_pairs
        load_size = load_size or self.load_size

        with torch.no_grad():
            points1, points2, image1_pil, image2_pil, patches_xy, desc1, desc2, num_patches = find_correspondences(
                image_1_path, image_2_path, num_pairs, load_size,
                self.layer, self.facet, self.bin, self.thresh,
                self.model_type, self.stride, return_patches_x_y=True
            )
            desc1 = desc1.reshape((num_patches[0], num_patches[1], -1))
            descriptor_vectors = desc1[patches_xy[0], patches_xy[1]]

            if self.visualize_matches:
                print("Visualizing matches...")
                fig_1, ax1 = plt.subplots()
                ax1.axis('off')
                ax1.imshow(image1_pil)
                fig_2, ax2 = plt.subplots()
                ax2.axis('off')
                ax2.imshow(image2_pil)
                fig1, fig2 = draw_correspondences(points1, points2, image1_pil, image2_pil)
                plt.show()
            return patches_xy, desc1, descriptor_vectors, num_patches

    def extract_desc_maps(self, image_paths, load_size=None):
        load_size = load_size or self.load_size
            
        if not isinstance(image_paths, list):
            image_paths = [image_paths]
        path = image_paths[0]
        if isinstance(path, str):
            pass
        else:
            paths = []
            for i in range(len(image_paths)):
                paths.append(f"image_{i}.png")
                media.write_image( f"image_{i}.png", image_paths[i])
            image_paths = paths

        descriptors_list = []
        org_images_list = []
        with torch.no_grad():
            for i, path in enumerate(image_paths):
                image_batch, _ = self.extractor.preprocess(path, load_size)
                descriptors = self.extractor.extract_descriptors(image_batch.to(self.device), self.layer, self.facet, self.bin)
                patched_shape = self.extractor.num_patches
                descriptors = descriptors.reshape((patched_shape[0], patched_shape[1], -1))
                descriptors_list.append(descriptors.cpu())

            img_np = image_batch[0].cpu().numpy()  # convert tensor to numpy
            img_np = np.transpose(img_np, (1, 2, 0))
            img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255).astype(np.uint8)
            org_images_list.append(media.resize_image(
                img_np, (img_np.shape[0] // self.patch_size, img_np.shape[1] // self.patch_size)
            ))

        # clean up temporary files if created
        if not isinstance(path, str):
            import os
            for p in image_paths:
                if os.path.exists(p):
                    os.remove(p)
        return descriptors_list, org_images_list

    def extract_descriptor_nn(self, descriptors, emb_im, patched_shape, return_heatmaps=False):
        cs_ys, cs_xs, cs_list = [], [], []
        cs = torch.nn.CosineSimilarity(dim=-1)
        for d in descriptors:
            cs_i = cs(d.cuda(), emb_im.cuda()).reshape(-1)
            y, x = cs_i.argmax().cpu() // patched_shape[1], cs_i.argmax().cpu() % patched_shape[1]
            cs_ys.append(int(y))
            cs_xs.append(int(x))
            cs_list.append(cs_i.cpu().reshape(patched_shape))
        return (cs_ys, cs_xs, cs_list) if return_heatmaps else (cs_ys, cs_xs)


    def run(self):
        pass

def main():
    """Test LightGlue visual servoing node."""
    rclpy.init()

    # Example usage
    DIR = "example_tasks/pan"
    rgb_ref = np.array(Image.open(f"{DIR}/ref_rgb_wrist.png"))
    seg_ref = np.array(Image.open(f"{DIR}/ref_mask_wrist.png")).astype(bool)

    servoer = DINOVisualServoer(
        DIR=DIR,
        rgb_ref=rgb_ref,
        seg_ref=seg_ref,
        use_depth=True,
        silent=False,
        visualize_matches=True,
    )

    while rclpy.ok():
        servoer.match_dino(filter_seg_ref=True)
        rclpy.spin_once(servoer)

if __name__ == "__main__":
    main() 

# Usage: python3 -m match_servoers.dino_servoer

### Replace the find_correspondence method in dino_vit_features.correspondences with the following for real-time visual servoing.

# def find_correspondences(
#     image_path1: str,
#     image_path2: str,
#     num_pairs: int = 10,
#     load_size: int = 224,
#     layer: int = 9,
#     facet: str = "key",
#     bin: bool = True,
#     thresh: float = 0.05,
#     model_type: str = "dino_vits8",
#     stride: int = 4,
#     return_patches_x_y: bool = True,
# ) -> Tuple[
#     List[Tuple[float, float]],
#     List[Tuple[float, float]],
#     Image.Image,
#     Image.Image,
#     List[np.ndarray],
#     torch.Tensor,
#     torch.Tensor,
#     Tuple[int, int],
# ]:
#     """
#     Find high-quality point correspondences between two images using DINO ViT features.

#     Returns:
#         points1: list of (y,x) coordinates in image 1
#         points2: list of (y,x) coordinates in image 2
#         image1_pil: preprocessed PIL image 1
#         image2_pil: preprocessed PIL image 2
#         patches_xy: [img1_y, img1_x, img2_y, img2_x] patch grid coordinates (optional)
#         descriptors1: tensor of image 1 descriptors
#         descriptors2: tensor of image 2 descriptors
#         num_patches1: (H, W) number of patches in image 1
#     """

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # --- 1. Extract descriptors ---
#     extractor = ViTExtractor(model_type, stride, device=device)

#     img1_batch, img1_pil = extractor.preprocess(image_path1, load_size)
#     desc1 = extractor.extract_descriptors(img1_batch.to(device), layer, facet, bin)
#     num_patches1 = extractor.num_patches

#     img2_batch, img2_pil = extractor.preprocess(image_path2, load_size)
#     desc2 = extractor.extract_descriptors(img2_batch.to(device), layer, facet, bin)
#     num_patches2 = extractor.num_patches

#     # --- 2. Saliency maps & foreground masks ---
#     sal1 = extractor.extract_saliency_maps(img1_batch.to(device))[0]
#     sal2 = extractor.extract_saliency_maps(img2_batch.to(device))[0]
#     fg_mask1 = sal1 > thresh
#     fg_mask2 = sal2 > thresh

#     # --- 3. Similarity & Best-Buddies ---
#     sims = chunk_cosine_sim(desc1, desc2)
#     sim_1, nn_1 = torch.max(sims, dim=-1)  # block2 closest to block1
#     sim_2, nn_2 = torch.max(sims, dim=-2)  # block1 closest to block2

#     sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0]
#     sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0]

#     idxs_img1 = torch.arange(num_patches1[0] * num_patches1[1], device=device)
#     bbs_mask = nn_2[nn_1] == idxs_img1

#     # Remove matches where either descriptor is background
#     fg_mask2_new_coors = nn_2[fg_mask2]
#     fg_mask2_mask_new_coors = torch.zeros_like(bbs_mask, dtype=torch.bool)
#     fg_mask2_mask_new_coors[fg_mask2_new_coors] = True

#     bbs_mask = torch.bitwise_and(bbs_mask, fg_mask1)
#     bbs_mask = torch.bitwise_and(bbs_mask, fg_mask2_mask_new_coors)

#     # --- 4. K-Means to get well-distributed pairs ---
#     bb_descs1 = desc1[0, 0, bbs_mask, :].cpu().numpy()
#     bb_descs2 = desc2[0, 0, nn_1[bbs_mask], :].cpu().numpy()

#     all_keys = np.concatenate((bb_descs1, bb_descs2), axis=1)
#     n_clusters = min(num_pairs, len(all_keys))
#     normed = all_keys / np.linalg.norm(all_keys, axis=1, keepdims=True)
#     kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(normed)

#     bb_cls_attn = (sal1[bbs_mask] + sal2[nn_1[bbs_mask]]) / 2
#     top_sim = np.full(n_clusters, -np.inf)
#     top_idx = np.full(n_clusters, -1, dtype=int)

#     for i, (label, rank) in enumerate(zip(kmeans.labels_, bb_cls_attn)):
#         if rank > top_sim[label]:
#             top_sim[label] = rank
#             top_idx[label] = i

#     mask_idx = torch.nonzero(bbs_mask, as_tuple=False).squeeze(1)[top_idx]
#     img1_idx_show = idxs_img1[mask_idx]
#     img2_idx_show = nn_1[mask_idx]

#     img1_y = (img1_idx_show // num_patches1[1]).cpu().numpy()
#     img1_x = (img1_idx_show % num_patches1[1]).cpu().numpy()
#     img2_y = (img2_idx_show // num_patches2[1]).cpu().numpy()
#     img2_x = (img2_idx_show % num_patches2[1]).cpu().numpy()

#     # --- 5. Convert to pixel coordinates ---
#     points1, points2 = [], []
#     for y1, x1, y2, x2 in zip(img1_y, img1_x, img2_y, img2_x):
#         x1_px = (x1 - 1) * extractor.stride[1] + extractor.stride[1] + extractor.p // 2
#         y1_px = (y1 - 1) * extractor.stride[0] + extractor.stride[0] + extractor.p // 2
#         x2_px = (x2 - 1) * extractor.stride[1] + extractor.stride[1] + extractor.p // 2
#         y2_px = (y2 - 1) * extractor.stride[0] + extractor.stride[0] + extractor.p // 2

#         points1.append((y1_px, x1_px))
#         points2.append((y2_px, x2_px))

#     patches_xy = [img1_y, img1_x, img2_y, img2_x] if return_patches_x_y else []

#     return points1, points2, img1_pil, img2_pil, patches_xy, desc1, desc2, num_patches1