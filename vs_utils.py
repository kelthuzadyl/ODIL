import numpy as np
import cv2
import poselib
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_random_state

def rot_mat_to_euler(rot_mat: np.ndarray) -> np.ndarray:
    r = R.from_matrix(rot_mat)
    return r.as_euler("xyz")

def rot_mat_to_quat(rot_mat: np.ndarray) -> np.ndarray:
    r = R.from_matrix(rot_mat)
    return r.as_quat()

def compose_homogeneous_matrix(xyz: np.ndarray, quat: np.ndarray) -> np.ndarray:
    rot = R.from_quat(quat).as_matrix()
    matrix = np.eye(4)
    matrix[:3, :3] = rot
    matrix[:3, 3] = xyz
    return matrix

def compose_homogeneous_matrix_euler(xyz: np.ndarray, euler: np.ndarray) -> np.ndarray:
    rot = R.from_euler('xyz', euler).as_matrix()
    matrix = np.eye(4)
    matrix[:3, :3] = rot
    matrix[:3, 3] = xyz
    return matrix

def state_to_transform(state):
    """Convert state vector [x, y, z, roll, pitch, yaw] to transformation matrix."""
    t = state[:3]
    r = R.from_euler('xyz', state[3:], degrees=False).as_matrix()
    T = np.eye(4)
    T[:3, :3] = r
    T[:3, 3] = t
    return T

def transform_to_state(T):
    """Convert transformation matrix to state vector [x, y, z, roll, pitch, yaw]."""
    t = T[:3, 3]
    r = R.from_matrix(T[:3, :3]).as_euler('xyz', degrees=False)
    return np.concatenate((t, r))

def euler_from_matrix(matrix):
    """Extracts the euler from a 4x4 homogeneous transformation matrix."""
    rotation_matrix = matrix[:3, :3]
    rotation = R.from_matrix(rotation_matrix)
    return rotation.as_euler("xyz")

def euler_from_quat(quat, seq='xyz'):
    return R.from_quat(quat).as_euler(seq)

def quat_from_euler(euler, seq='xyz'):
    return R.from_euler('xyz', euler, degrees=False).as_quat()

def normalize_mkpts(mkpts, K):
    """
    Normalize image points using the camera intrinsic matrix.

    Args:
    mkpts: Nx2 numpy array of image points.
    K: 3x3 intrinsic matrix.

    Returns:
    normalized_points: Nx3 numpy array of normalized points in homogeneous coordinates.
    """
    # Convert points to homogeneous coordinates
    mkpts_h = np.hstack((mkpts, np.ones((mkpts.shape[0], 1))))

    # Normalize points
    normalized_points = (np.linalg.inv(K) @ mkpts_h.T).T

    return normalized_points

def add_depth(mkpts, depth, K):
    """
    Converts 2D points to 3D points using depth information.

    Parameters:
    mkpts (np.ndarray): Array of 2D points with shape (N, 2), where N is the number of points.
    depth (np.ndarray): Depth map with shape (H, W), where H and W are the height and width of the depth map.
    K (np.ndarray): Camera intrinsic matrix with shape (3, 3).

    Returns:
    np.ndarray: Array of 3D points with shape (N, 3).
    """
    normalized_mkpts = normalize_mkpts(mkpts, K)
    depth_values = depth[mkpts[:, 1].astype(int), mkpts[:, 0].astype(int)]
    points_3ds = normalized_mkpts * depth_values[:, np.newaxis] / 1000
    
    return points_3ds

def weighted_add_depth(mkpts_scores, depth, K):
    mkpts = mkpts_scores[:, :2]
    scores = mkpts_scores[:, 2]
    
    # Normalize the 2D points using the camera intrinsics
    normalized_mkpts = normalize_mkpts(mkpts, K)
    
    # Fetch the depth values corresponding to the 2D points
    depth_values = depth[mkpts[:, 1].astype(int), mkpts[:, 0].astype(int)]
    
    # Calculate the 3D points from the normalized points and depth values
    points_3d = normalized_mkpts * depth_values[:, np.newaxis] / 1000  # Scale to meters
    
    # Combine the 3D points with their corresponding scores
    points_3d_weighted = np.hstack([points_3d, scores[:, np.newaxis]])
    
    return points_3d_weighted

def compute_transform_least_square(X, Y):
    """
    Find the optimal rigid transformation (rotation and translation) between two sets of 3D points with least squares.

    This function calculates the rotation matrix and translation vector that align two sets of 3D points
    by minimizing the mean squared error between the transformed points in set X and the corresponding points in set Y.

    The transformation is computed using the following steps:
    1. Remove zero vectors from both point sets.
    2. Compute the centroids of the remaining points in each set.
    3. Center the points by subtracting their respective centroids.
    4. Compute the covariance matrix between the centered point sets.
    5. Perform Singular Value Decomposition (SVD) on the covariance matrix to obtain the rotation matrix.
    6. Compute the translation vector.

    Parameters:
    X (numpy.ndarray): A 2D array of shape (N, 3) representing the first set of 3D points.
    Y (numpy.ndarray): A 2D array of shape (N, 3) representing the second set of 3D points.

    Returns:
    R (numpy.ndarray): A 2D array of shape (3, 3) representing the rotation matrix.
    t (numpy.ndarray): A 1D array of shape (3,) representing the translation vector.

    Raises:
    ValueError: If the input arrays X and Y do not have the same shape or do not have three columns.
    """
    if X.shape != Y.shape or X.shape[1] != 3:
        raise ValueError("Input arrays X and Y must have the same shape and must be 2D arrays with 3 columns.")

    # Calculate centroids
    cX = np.mean(X, axis=0)
    cY = np.mean(Y, axis=0)
    
    # Subtract centroids to obtain centered sets of points
    Xc = X - cX
    Yc = Y - cY
    
    # Calculate covariance matrix
    C = np.dot(Xc.T, Yc)
    
    # Compute SVD
    U, S, Vt = np.linalg.svd(C)
    
    # Determine rotation matrix
    R = np.dot(Vt.T, U.T)
    
    # Ensure a right-handed coordinate system
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    
    # Determine translation vector
    t = cY - np.dot(R, cX)

    return R, t

def weighted_compute_transform_least_square(X_weighted, Y_weighted):
    """
    Find the optimal rigid transformation (rotation and translation) between two sets of 3D points with weighted least squares.

    This function calculates the rotation matrix and translation vector that align two sets of 3D points
    by minimizing the weighted mean squared error between the transformed points in set X and the corresponding points in set Y.
    
    The weights are derived from the scores associated with the points.

    Parameters:
    X_weighted (numpy.ndarray): A 2D array of shape (N, 4) representing the first set of 3D points with associated scores [X, Y, Z, score].
    Y_weighted (numpy.ndarray): A 2D array of shape (N, 4) representing the second set of 3D points with associated scores [X, Y, Z, score].

    Returns:
    R (numpy.ndarray): A 2D array of shape (3, 3) representing the rotation matrix.
    t (numpy.ndarray): A 1D array of shape (3,) representing the translation vector.

    Raises:
    ValueError: If the input arrays X_weighted and Y_weighted do not have the same shape or do not have four columns.
    """
    if X_weighted.shape != Y_weighted.shape or X_weighted.shape[1] != 4:
        raise ValueError("Input arrays X_weighted and Y_weighted must have the same shape and must be 2D arrays with 4 columns.")

    # Separate the points and scores
    X = X_weighted[:, :3]
    Y = Y_weighted[:, :3]
    weights = X_weighted[:, 3]  # Use the scores as weights

    # Normalize weights
    weights = weights / np.sum(weights)

    # Calculate weighted centroids
    cX = np.average(X, axis=0, weights=weights)
    cY = np.average(Y, axis=0, weights=weights)
    
    # Subtract centroids to obtain centered sets of points
    Xc = X - cX
    Yc = Y - cY
    
    # Calculate weighted covariance matrix
    C = np.dot((weights[:, np.newaxis] * Xc).T, Yc)
    
    # Compute SVD
    U, S, Vt = np.linalg.svd(C)
    
    # Determine rotation matrix
    R = np.dot(Vt.T, U.T)
    
    # Ensure a right-handed coordinate system
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    
    # Determine translation vector
    t = cY - np.dot(R, cX)

    return R, t

def ransac_find_transformation(X, Y, max_iterations=100, threshold=0.005, random_state=None, fall_back=True):
    """
    Find the optimal rigid transformation (rotation and translation) between two sets of 3D points using RANSAC.

    Parameters:
    X (numpy.ndarray): A 2D array of shape (N, 3) representing the first set of 3D points.
    Y (numpy.ndarray): A 2D array of shape (N, 3) representing the second set of 3D points.
    max_iterations (int): Maximum number of RANSAC iterations.
    threshold (float): Inlier threshold for RANSAC.
    random_state (int or None): Seed for random number generator.
    adaptive (bool): Whether to use an adaptive threshold if no transformation is found.

    Returns:
    R (numpy.ndarray or None): A 2D array of shape (3, 3) representing the rotation matrix, or None if not enough points.
    t (numpy.ndarray or None): A 1D array of shape (3,) representing the translation vector, or None if not enough points.
    inliers (numpy.ndarray or None): A 1D boolean array indicating which points are inliers, or None if not enough points.
    """
    if X.shape != Y.shape or X.shape[1] != 3:
        raise ValueError("Input arrays X and Y must have the same shape and must be 2D arrays with 3 columns.")
    
    num_points = X.shape[0]
    
    if num_points < 3:
        # Not enough points to define a transformation
        print("Not enough points to compute a transformation.")
        return None, None

    rng = check_random_state(random_state)
    best_inliers = None
    best_R = None
    best_t = None
    best_error = float('inf')

    min_points = 4 if num_points >= 4 else num_points

    for i in range(max_iterations):
        # Randomly sample min_points points (or fewer if less than 4 points available)
        indices = rng.choice(num_points, size=min_points, replace=False)
        X_sample = X[indices]
        Y_sample = Y[indices]

        # Skip iteration if we somehow end up with empty samples
        if X_sample.shape[0] == 0 or Y_sample.shape[0] == 0:
            continue

        # Calculate transformation for the sample
        R, t = compute_transform_least_square(X_sample, Y_sample)

        if R is None or t is None:
            continue  # Skip if no valid transformation was computed

        # Transform all points using the estimated rotation and translation
        Y_pred = np.dot(X, R.T) + t

        # Compute inliers based on threshold
        errors = np.linalg.norm(Y_pred - Y, axis=1)
        inliers = errors < threshold

        # Skip if no inliers found
        if np.sum(inliers) == 0:
            continue

        # Check if current model is better
        current_error = mean_squared_error(Y[inliers], Y_pred[inliers])
        if best_inliers is None or np.sum(inliers) > np.sum(best_inliers) or (np.sum(inliers) == np.sum(best_inliers) and current_error < best_error):
            best_inliers = inliers
            best_R = R
            best_t = t
            best_error = current_error

    if best_R is None or best_t is None:
        print("RANSAC failed to find a valid transformation with the initial threshold.")

        if not fall_back:
            return None, None
        
        # Fallback mechanism: using all points
        else:
            print("Fallback: Computing transformation using all points.")
            R, t = compute_transform_least_square(X, Y)
            return R, t

    return best_R, best_t

def solve_transform_3d(mkpts_0, mkpts_1, depth_ref, depth_cur, K_ref, K_cur=None, fall_back=True):
    """
    Compute the 3D transformation matrix between two sets of 3D points derived from depth maps.

    This function estimates the transformation between two sets of 3D points (from depth maps) by:
    1. Converting 2D keypoints and depth maps into 3D points.
    2. Calculating the rigid transformation (rotation and translation) between the two sets of 3D points.
    3. Constructing a 4x4 transformation matrix from the computed rotation and translation.

    Parameters:
    mkpts_0 (numpy.ndarray): A 2D array of shape (N, 2) containing 2D keypoints from the reference image.
    mkpts_1 (numpy.ndarray): A 2D array of shape (N, 2) containing 2D keypoints from the current image.
    depth_ref (numpy.ndarray): A 2D array representing the depth map of the reference image.
    depth_cur (numpy.ndarray): A 2D array representing the depth map of the current image.

    Returns:
    T_est (numpy.ndarray): A 4x4 transformation matrix that includes the rotation and translation between the two sets of 3D points.

    Raises:
    ValueError: If the shapes of `mkpts_0` and `mkpts_1` do not match or if the depth maps and keypoints are incompatible.
    """

    if K_cur is None:
        # Convert 2D keypoints and depth maps to 3D points
        mkpts_0_3d = add_depth(mkpts_0, depth_ref, K_ref)
        mkpts_1_3d = add_depth(mkpts_1, depth_cur, K_ref)
    else:
        mkpts_0_3d = add_depth(mkpts_0, depth_ref, K_ref)
        mkpts_1_3d = add_depth(mkpts_1, depth_cur, K_cur)
        
    non_zero_indices = np.all(mkpts_0_3d != [0, 0, 0], axis=1) & np.all(mkpts_1_3d != [0, 0, 0], axis=1)
    mkpts_0_3d = mkpts_0_3d[non_zero_indices]
    mkpts_1_3d = mkpts_1_3d[non_zero_indices]

    # Compute the transformation between the two sets of 3D points
    delta_R_camera, delta_t_camera = ransac_find_transformation(mkpts_0_3d, mkpts_1_3d, fall_back=fall_back)

    if delta_R_camera is None or delta_t_camera is None:
        return None
    
    # Create a 4x4 transformation matrix
    T_est = np.eye(4)
    T_est[:3, :3] = delta_R_camera
    T_est[:3, 3] = delta_t_camera
    
    return T_est

def weighted_solve_transform_3d(mkpts_scores_0, mkpts_scores_1, depth_ref, depth_cur, K):

    # Convert 2D keypoints and depth maps to 3D points
    mkpts_0_3d_weighted = weighted_add_depth(mkpts_scores_0, depth_ref, K)
    mkpts_1_3d_weighted = weighted_add_depth(mkpts_scores_1, depth_cur, K)
    
    non_zero_indices = (mkpts_0_3d_weighted[:, 2] != 0) & (mkpts_1_3d_weighted[:, 2] != 0)
    mkpts_0_3d_weighted = mkpts_0_3d_weighted[non_zero_indices]
    mkpts_1_3d_weighted = mkpts_1_3d_weighted[non_zero_indices]

    # Compute the transformation between the two sets of 3D points
    delta_R_camera, delta_t_camera = weighted_compute_transform_least_square(mkpts_0_3d_weighted, mkpts_1_3d_weighted)

    # Create a 4x4 transformation matrix
    T_est = np.eye(4)
    T_est[:3, :3] = delta_R_camera
    T_est[:3, 3] = delta_t_camera

    return mkpts_0_3d_weighted, mkpts_1_3d_weighted, T_est

def model_selection_homography(mkpts_0, mkpts_1, K, W, H, H_inlier_ratio_thr=0.5):
    # Normalize keypoints using camera intrinsics
    normalized_mkpts_0 = normalize_mkpts(mkpts_0, K)
    normalized_mkpts_1 = normalize_mkpts(mkpts_1, K)

    # Compute homography using normalized keypoints
    H_norm, mask = cv2.findHomography(normalized_mkpts_0, normalized_mkpts_1, cv2.USAC_MAGSAC, 0.005, confidence=0.99999)
    H_inlier_ratio = np.sum(mask) / mask.shape[0]
    
    camera = {'model': 'PINHOLE', 'width': W, 'height': H, 'params': [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]}
    M, info = poselib.estimate_relative_pose(mkpts_0, mkpts_1, camera, camera, {"max_epipolar_error": 0.5})
    E_inlier_ratio = info['num_inliers'] / mask.shape[0]

    # Print the decision criteria and the final decision
    decision = not (E_inlier_ratio > 1.5 * H_inlier_ratio and H_inlier_ratio < H_inlier_ratio_thr)
    print(f"E_inlier_ratio: {E_inlier_ratio:.3f}, H_inlier_ratio: {H_inlier_ratio:.3f}, is dominated by planes: {decision}")
    
    return decision