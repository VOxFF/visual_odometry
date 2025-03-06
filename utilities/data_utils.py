import pandas as pd
import numpy as np


def read_ground_truth_positions(gt_file_path, skip=1):
    """
    Reads the ground truth file and returns an array of global positions,
    skipping rows according to the provided skip value.

    Expected file format:
        # timestamp tx ty tz qx qy qz qw
        1540821844.73095 7.04218 3.36935 -0.91425 0.00384 -0.00320 0.89610 -0.44382
        1540821844.73295 7.04216 3.36934 -0.91422 0.00386 -0.00321 0.89610 -0.44381
        ...

    The function ignores comment lines (starting with '#').

    Args:
        gt_file_path (str): Path to the ground truth file.
        skip (int): Take every nth row (default is 1, meaning no skipping).

    Returns:
        np.ndarray: Array of shape (N, 3) where each row is [tx, ty, tz].
    """
    try:
        df_gt = pd.read_csv(gt_file_path, delim_whitespace=True, comment='#',
                            names=['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
    except Exception as e:
        print(f"Error reading ground truth file: {e}")
        return None

    df_gt.sort_values('timestamp', inplace=True)
    # Skip rows according to the skip parameter.
    positions = df_gt[['tx', 'ty', 'tz']].to_numpy()[::skip]
    print(f"Read {positions.shape[0]} ground truth positions from {gt_file_path} (using skip={skip}).")
    return positions

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """
    Converts a quaternion to a 3x3 rotation matrix.

    Args:
        qx, qy, qz, qw (float): Quaternion components.

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    # Normalize the quaternion to avoid scaling issues.
    norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    R = np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)]
    ])
    return R


def read_ground_truth_transforms(gt_file_path, skip=1):
    """
    Reads the ground truth file and returns a list of 4x4 transformation matrices,
    skipping rows according to the provided skip value.

    Expected file format:
        # timestamp tx ty tz qx qy qz qw
        1540821844.73095 7.04218 3.36935 -0.91425 0.00384 -0.00320 0.89610 -0.44382
        1540821844.73295 7.04216 3.36934 -0.91422 0.00386 -0.00321 0.89610 -0.44381
        ...

    Returns:
        list: A list of 4x4 numpy arrays where each matrix is constructed as:
              [ R  t ]
              [ 0  1 ]
              with R computed from the quaternion.
    """
    df_gt = pd.read_csv(gt_file_path, delim_whitespace=True, comment='#',
                        names=['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
    df_gt.sort_values('timestamp', inplace=True)
    # Skip rows according to the skip parameter.
    df_gt = df_gt.iloc[::skip]

    transforms = []
    for idx, row in df_gt.iterrows():
        t = np.array([row['tx'], row['ty'], row['tz']])
        R = quaternion_to_rotation_matrix(row['qx'], row['qy'], row['qz'], row['qw'])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        transforms.append(T)

    print(f"Read {len(transforms)} ground truth transformation matrices from {gt_file_path} (using skip={skip}).")
    return transforms

def match_ground_truth_positions(computed_positions, image_list_path, gt_file_path, tolerance=0.05):
    """
    Matches computed global positions with ground truth positions based on timestamps.

    Args:
        computed_positions (list or np.ndarray): List/array of computed global positions.
            It is assumed that computed_positions[i] corresponds to the image at row i in the image list file.
        image_list_path (str): Path to the image list file.
            Expected format: "# id timestamp image_name" (whitespace‑delimited).
        gt_file_path (str): Path to the ground truth file.
            Expected format: "# timestamp tx ty tz qx qy qz qw" (whitespace‑delimited).
        tolerance (float): Maximum allowed difference (in seconds) between image and GT timestamps.

    Returns:
        list of tuples: Each tuple is (computed_position, gt_position)
            where gt_position is a 3-element array [tx, ty, tz].
            If no ground truth is found within the tolerance for a computed position,
            gt_position will be set to None.
    """
    # Read the image list file.
    df_images = pd.read_csv(image_list_path, delim_whitespace=True, comment='#',
                            names=['id', 'timestamp', 'image_name'])
    df_images.sort_values('timestamp', inplace=True)

    # Read the ground truth file.
    df_gt = pd.read_csv(gt_file_path, delim_whitespace=True, comment='#',
                        names=['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
    df_gt.sort_values('timestamp', inplace=True)

    results = []
    for i, comp_pos in enumerate(computed_positions):
        try:
            img_ts = float(df_images.iloc[i]['timestamp'])
        except IndexError:
            print(f"Index {i}: No image timestamp found. Setting ground truth to None.")
            results.append((np.array(comp_pos), np.array([0, 0, 0])))
            continue

        # Calculate the absolute time differences.
        time_diffs = np.abs(df_gt['timestamp'] - img_ts)
        min_diff = time_diffs.min()
        idx_closest = time_diffs.idxmin()

        print(f"{i} min diff: {min_diff}");

        if min_diff <= tolerance:
            gt_row = df_gt.loc[idx_closest]
            gt_position = np.array([gt_row['tx'], gt_row['ty'], gt_row['tz']])
            print(f"Index {i}: Image timestamp {img_ts:.3f} matched with GT timestamp {df_gt.loc[idx_closest]['timestamp']:.3f} (diff: {min_diff:.3f}s)")
        else:
            gt_position = np.array([0, 0, 0])
            print(f"Index {i}: No GT within tolerance for image timestamp {img_ts:.3f} (min diff: {min_diff:.3f}s > tol {tolerance})")

        results.append((np.array(comp_pos), gt_position))

    print(f"matches found: {len(results)}")



    return results
