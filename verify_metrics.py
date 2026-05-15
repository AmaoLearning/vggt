import sys
import numpy as np
from scipy.spatial.transform import Rotation
import copy
from evo.core.trajectory import PoseTrajectory3D
from evo.core.metrics import PoseRelation, Unit
from evo.main_ape import ape
from evo.main_rpe import rpe

def compute_metrics_fixed(pred_extri, gt_poses_c2w):
    num_poses = len(pred_extri)
    timestamps = np.arange(num_poses)
    gt_w2c = np.linalg.inv(gt_poses_c2w)
    traj_ref = PoseTrajectory3D(
        positions_xyz=gt_w2c[:, :3, 3],
        orientations_quat_wxyz=Rotation.from_matrix(gt_w2c[:, :3, :3]).as_quat(scalar_first=True),
        timestamps=timestamps
    )
    pred_w2c = np.zeros((num_poses, 4, 4))
    pred_w2c[:, :3, :] = pred_extri
    pred_w2c[:, 3, 3] = 1.0
    traj_est = PoseTrajectory3D(
        positions_xyz=pred_w2c[:, :3, 3],
        orientations_quat_wxyz=Rotation.from_matrix(pred_w2c[:, :3, :3]).as_quat(scalar_first=True),
        timestamps=timestamps
    )
    ate_res = ape(copy.deepcopy(traj_ref), copy.deepcopy(traj_est), est_name="traj", 
                  pose_relation=PoseRelation.translation_part, align=False, correct_scale=False)
    rpe_res = rpe(copy.deepcopy(traj_ref), copy.deepcopy(traj_est), est_name="traj", 
                  pose_relation=PoseRelation.translation_part, delta=1, delta_unit=Unit.frames, 
                  align=False, correct_scale=False)
    return ate_res, rpe_res

pred_extri = np.zeros((3, 3, 4))
for i in range(3):
    pred_extri[i, :3, :3] = np.eye(3)
pred_extri[0, :3, 3] = [0, 0, 0]
pred_extri[1, :3, 3] = [1, 0, 0]
pred_extri[2, :3, 3] = [0, 1, 0]

gt_poses_c2w = np.zeros((3, 4, 4))
for i in range(3):
    gt_poses_c2w[i] = np.eye(4)
    gt_poses_c2w[i, :3, 3] = -pred_extri[i, :3, 3]

ate_res, rpe_res = compute_metrics_fixed(pred_extri, gt_poses_c2w)
ate = ate_res.stats["mean"]
rpe_t = rpe_res.stats["mean"]
print(f"ATE: {ate}")
print(f"RPE_trans: {rpe_t}")
print(f"Near-zero: {ate < 1e-6 and rpe_t < 1e-6}")
