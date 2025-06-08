import argparse
from typing import Optional, Tuple, List
import numpy as np
import cv2
from pyapriltags import Detector

from src.type import Grasp
from src.utils import to_pose
from src.sim.wrapper_env import WrapperEnvConfig, WrapperEnv
from src.sim.wrapper_env import get_grasps
from src.test.load_test import load_test_data

from pose_est.config import Pose_Est_Config
from pose_est.utils import get_exp_config_from_checkpoint, farthest_point_sampling, get_pc
from pose_est.model import get_pose_est_model
import torch, time

DEPTH_IMG_SCALE = 16384

TABLE_HEIGHT = 0.66
OBJ_INIT_TRANS = np.array([0.51, 0.365, 0.82])
OBJ_RAND_RANGE = 0.14

PC_MIN = np.array(
    [
        OBJ_INIT_TRANS[0] - OBJ_RAND_RANGE / 2,
        OBJ_INIT_TRANS[1] - OBJ_RAND_RANGE /2,
        0.69,
    ]
)
PC_MAX = np.array(
    [
        OBJ_INIT_TRANS[0] + OBJ_RAND_RANGE / 2,
        OBJ_INIT_TRANS[1] + OBJ_RAND_RANGE /2,
        0.895,
    ]
)

def transform_obj_pose_to_world(
    obj_trans_cam: np.ndarray,  # 物体在相机坐标系的平移 [3,]
    obj_rot_cam: np.ndarray,    # 物体在相机坐标系的旋转 [3, 3]
    cam_trans_world: np.ndarray, # 相机在世界坐标系的平移 [3,]
    cam_rot_world: np.ndarray    # 相机在世界坐标系的旋转 [3, 3]
) -> np.ndarray:
    """
    将物体位姿从相机坐标系转换到世界坐标系。

    参数:
        obj_trans_cam: 物体在相机坐标系的平移向量。
        obj_rot_cam: 物体在相机坐标系的旋转矩阵。
        cam_trans_world: 相机在世界坐标系的平移向量。
        cam_rot_world: 相机在世界坐标系的旋转矩阵。

    返回:
        obj_pose_world: 物体在世界坐标系的4x4位姿矩阵。
    """
    # 1. 物体坐标系 → 相机坐标系（如果obj_rot/trans已经是相机坐标系下的，可跳过）
    # （此处假设obj_rot_cam和obj_trans_cam已经是相机坐标系下的值，无需额外变换）

    # 2. 相机坐标系 → 世界坐标系
    obj_trans_world = cam_rot_world @ obj_trans_cam + cam_trans_world
    obj_rot_world = cam_rot_world @ obj_rot_cam

    # 组合为4x4位姿矩阵
    obj_pose_world = np.eye(4)
    obj_pose_world[:3, :3] = obj_rot_world
    obj_pose_world[:3, 3] = obj_trans_world
    obj_pose_world[2,3]-=0.04  # 调整Z轴位置，确保物体在桌面上方

    return obj_pose_world

import open3d as o3d
def visualize_point_cloud(points, title="Point Cloud"):
    """
    可视化点云（兼容所有Open3D版本）
    """
    # 转换为numpy数组
    if hasattr(points, 'cpu'):
        points = points.cpu().numpy()
    if points.ndim == 3:
        points = points.squeeze(0)
    
    # 计算并打印坐标信息
    min_pt = np.min(points, axis=0)
    max_pt = np.max(points, axis=0)
    print(f"\n{title} 坐标范围:")
    print(f"X: {min_pt[0]:.2f} → {max_pt[0]:.2f}")
    print(f"Y: {min_pt[1]:.2f} → {max_pt[1]:.2f}")
    print(f"Z: {min_pt[2]:.2f} → {max_pt[2]:.2f}")
    
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0, 0, 1])
    
    # 创建坐标系（自适应大小）
    coord_size = np.ptp(points, axis=0).max()/5
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=coord_size,
        origin=[0, 0, 0]
    )
    
    # 可视化
    o3d.visualization.draw_geometries([pcd, coord_frame], window_name=title)

def get_workspace_mask(pc: np.ndarray, z: float) -> np.ndarray:
    """Get the mask of the point cloud in the workspace."""
    z_min = z
    z_max = z + 0.1
    pc_mask = (
        (pc[:, 0] > PC_MIN[0])
        & (pc[:, 0] < PC_MAX[0])
        & (pc[:, 1] > PC_MIN[1])
        & (pc[:, 1] < PC_MAX[1])
        & (pc[:, 2] > z_min)
        & (pc[:, 2] < z_max)
    )
    return pc_mask

def detect_driller_pose(img, depth, camera_matrix, camera_pose, model, *args, **kwargs):
    """
    Detects the pose of driller, you can include your policy in args
    """
    # implement the detection logic here

    full_pc_camera = get_pc(depth, camera_matrix) * np.array([-1, -1, 1])
    print(depth[170][520])
    
    full_pc_world = (
                np.einsum("ab,nb->na", camera_pose[:3, :3], full_pc_camera)
                + camera_pose[:3, 3]
            )

    point_num = 1024  
    pc_mask = get_workspace_mask(full_pc_world, 0.6)
    # for z in range(75, 68, -1):
    #     height = z / 100
    #     pc_mask = get_workspace_mask(full_pc_world, height)
    #     if np.sum(pc_mask) > point_num:
    #         print(f"Height {height} has enough points: {np.sum(pc_mask)}")
    #         break
    sel_pc_idx = np.random.randint(0, np.sum(pc_mask), point_num)
    pc_camera = torch.tensor(full_pc_camera[pc_mask][sel_pc_idx], dtype=torch.float32).unsqueeze(0).to("cuda:0")  # (1, N, 3)
    
    visualize_point_cloud(pc_camera, "Sampled Point Cloud (World Frame) - After Masking")    
    trans_pred, rot_pred = model.est(pc_camera) 
    
    obj_pose = np.eye(4)
    obj_pose[:3, :3] = rot_pred.detach().cpu().numpy().copy()
    obj_pose[:3, 3] = trans_pred.detach().cpu().numpy().copy()
    
    # 提取相机在世界坐标系的位姿
    cam_rot_world = camera_pose[:3, :3]  # [3, 3]
    cam_trans_world = camera_pose[:3, 3] # [3,]

    # 转换到世界坐标系
    obj_pose_world = transform_obj_pose_to_world(
        obj_trans_cam=obj_pose[:3, 3],
        obj_rot_cam=obj_pose[:3, :3],
        cam_trans_world=cam_trans_world,
        cam_rot_world=cam_rot_world
    )
    
    return obj_pose_world

def detect_marker_pose(
        detector: Detector, 
        img: np.ndarray, 
        camera_params: tuple, 
        camera_pose: np.ndarray,
        tag_size: float = 0.12
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    # implement
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)
    if len(tags) == 0:
        return None, None
    tag = tags[0]
    # print(f"T: {tag.pose_t}\nR: {tag.pose_R}")
    trans_marker_camera = tag.pose_t.flatten()
    rot_marker_camera = tag.pose_R
    T_marker_cam = np.eye(4)
    T_marker_cam[:3, :3] = rot_marker_camera
    T_marker_cam[:3, 3] = trans_marker_camera
    
    T_marker_world = camera_pose @ T_marker_cam
    
    trans_marker_world = T_marker_world[:3, 3]
    rot_marker_world = T_marker_world[:3, :3]
    
    return trans_marker_world, rot_marker_world

def get_yaw_from_pose(pose):
    """从4x4的齐次矩阵中提取yaw角（绕Z轴旋转）"""
    # 取旋转矩阵部分
    rot_mat = pose[:3, :3]
    yaw = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
    return yaw

def forward_quad_policy(pose, target_pose, target_yaw, *args, **kwargs):
    """ guide the quadruped to position where you drop the driller """
    # implement
    pos = pose[:3, 3]
    target = target_pose[:3, 3]
    delta = target - pos
    direction = delta[:2]
    action_xy = np.clip(direction, -0.2, 0.2)

    current_yaw = get_yaw_from_pose(pose)
    delta_yaw =  target_yaw - current_yaw  
    action_yaw = np.clip(delta_yaw, -0.4, 0.4) if abs(delta_yaw) > 0.05 else 0.0

    if current_yaw < 0:
        action_xy *= -1

    return np.array([action_xy[0], action_xy[1], action_yaw]) if (current_yaw < -0.25 * target_yaw or current_yaw > 0.9 * target_yaw) else np.array([0, 0, action_yaw])

def backward_quad_policy(pose, target_pose, target_yaw, *args, **kwargs):
    """ guide the quadruped back to its initial position """
    # implement
    pos = pose[:3, 3]
    target = target_pose[:3, 3]
    delta = target - pos
    direction = delta[:2]
    action_xy = np.clip(direction, -0.2, 0.2)

    current_yaw = get_yaw_from_pose(pose)
    delta_yaw =  target_yaw - current_yaw  
    # print(f"Current Yaw: {current_yaw}, Target Yaw: {target_yaw}, Delta Yaw: {delta_yaw}")
    action_yaw = np.clip(delta_yaw, -0.4, 0.4) if abs(delta_yaw) > 0.05 else 0.0
    if current_yaw < 0:
        action_xy *= -1
        
    if pos[0] < 0.85 and current_yaw > -0.25 * target_yaw:    
        return np.array([action_xy[0], action_xy[1], 0])

    return np.array([action_xy[0], action_xy[1], action_yaw]) if (current_yaw < 0.9 * target_yaw) else np.array([0, 0, action_yaw])

def plan_grasp(env: WrapperEnv, grasp: Grasp, grasp_config, *args, **kwargs) -> Optional[List[np.ndarray]]:
    """Try to plan a grasp trajectory for the given grasp. The trajectory is a list of joint positions. Return None if the trajectory is not valid."""
    # implement
    reach_steps = grasp_config['reach_steps']
    lift_steps = grasp_config['lift_steps']
    delta_dist = grasp_config['delta_dist']

    traj_reach = []
    traj_lift = []

    _, depth = env.humanoid_robot_cfg.gripper_width_to_angle_depth(grasp.width)
    grasp_trans = grasp.trans - depth * grasp.rot[:, 0]
    succ, grasp_arm_qpos = env.humanoid_robot_model.ik(grasp_trans, grasp.rot)
    if not succ:
        return None

    traj_reach = [grasp_arm_qpos]
    traj_lift = [grasp_arm_qpos]

    cur_trans, cur_rot, cur_qpos = (
        grasp_trans.copy(),
        grasp.rot.copy(),
        grasp_arm_qpos.copy(),
    )
    for _ in range(reach_steps):
        cur_trans = cur_trans - delta_dist * cur_rot[:, 0]
        succ, cur_qpos = env.humanoid_robot_model.ik(
            cur_trans, cur_rot, cur_qpos, delta_thresh=0.5
        )
        if not succ:
            return None
        traj_reach = [cur_qpos] + traj_reach

    cur_trans, cur_rot, cur_qpos = (
        grasp_trans.copy(),
        grasp.rot.copy(),
        grasp_arm_qpos.copy(),
    )
    for _ in range(lift_steps):
        cur_trans[2] += delta_dist
        succ, cur_qpos = env.humanoid_robot_model.ik(
            cur_trans, cur_rot, cur_qpos, delta_thresh=0.5
        )
        if not succ:
            return None
        traj_lift.append(cur_qpos)

    return [np.array(traj_reach), np.array(traj_lift)]

def plan_move(env, begin_qpos, begin_trans, begin_rot, end_trans, end_rot, steps=100, *args, **kwargs):
    traj = []
    cur_qpos = begin_qpos.copy()

    cur_trans = begin_trans.copy()
    cur_rot = begin_rot.copy()

    delta_trans = (end_trans - begin_trans) / steps
    for _ in range(steps):
        cur_trans = cur_trans + delta_trans
        succ, cur_qpos = env.humanoid_robot_model.ik(
            cur_trans, cur_rot, cur_qpos, delta_thresh=1
        )
        if not succ:
            return None
        traj.append(cur_qpos.copy())

    return np.array(traj)


def open_gripper(env: WrapperEnv, steps = 10):
    for _ in range(steps):
        env.step_env(gripper_open=1)
def close_gripper(env: WrapperEnv, steps = 10):
    for _ in range(steps):
        env.step_env(gripper_open=0)
def plan_move_qpos(begin_qpos, end_qpos, steps=50) -> np.ndarray:
    delta_qpos = (end_qpos - begin_qpos) / steps
    cur_qpos = begin_qpos.copy()
    traj = []
    
    for _ in range(steps):
        cur_qpos += delta_qpos
        traj.append(cur_qpos.copy())
    
    return np.array(traj)
def execute_plan(env: WrapperEnv, plan):
    """Execute the plan in the environment."""
    for step in range(len(plan)):
        env.step_env(
            humanoid_action=plan[step],
        )


TESTING = True
DISABLE_GRASP = False
DISABLE_MOVE = True

def main():
    parser = argparse.ArgumentParser(description="Launcher config - Physics")
    parser.add_argument("--robot", type=str, default="galbot")
    parser.add_argument("--obj", type=str, default="power_drill")
    parser.add_argument("--ctrl_dt", type=float, default=0.02)
    parser.add_argument("--headless", type=int, default=0)
    parser.add_argument("--reset_wait_steps", type=int, default=100)
    parser.add_argument("--test_id", type=int, default=0)

    args = parser.parse_args()

    detector = Detector(
        families="tagStandard52h13",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )

    env_config = WrapperEnvConfig(
        humanoid_robot=args.robot,
        obj_name=args.obj,
        headless=args.headless,
        ctrl_dt=args.ctrl_dt,
        reset_wait_steps=args.reset_wait_steps,
    )


    env = WrapperEnv(env_config)
    if TESTING:
        data_dict = load_test_data(args.test_id)
        env.set_table_obj_config(
            table_pose=data_dict['table_pose'],
            table_size=data_dict['table_size'],
            obj_pose=data_dict['obj_pose']
        )
        env.set_quad_reset_pos(data_dict['quad_reset_pos'])

    env.launch()
    env.reset(humanoid_qpos=env.sim.humanoid_robot_cfg.joint_init_qpos)
    humanoid_init_qpos = env.sim.humanoid_robot_cfg.joint_init_qpos[:7]
    Metric = {
        'obj_pose': False,
        'drop_precision': False,
        'quad_return': False,
    }
    
    head_init_qpos = np.array([0.0,0.4]) # you can adjust the head init qpos to find the driller

    env.step_env(humanoid_head_qpos=head_init_qpos)
    
    observing_qpos = humanoid_init_qpos + np.array([-0.2,-0.1,0.3,0,0.1,0.3,0]) # you can customize observing qpos to get wrist obs
    # observing_qpos = humanoid_init_qpos + np.array([0, -0.1, 0, 0, 0, 0, 0]) # you can customize observing qpos to get wrist obs
    
    
    init_plan = plan_move_qpos(humanoid_init_qpos, observing_qpos, steps = 50)
    execute_plan(env, init_plan)
    
    # eef = env.humanoid_robot_model.fk_eef(observing_qpos)
    # print(eef)
    
    obs_head = env.get_obs(camera_id=0) # head camera
    obs_wrist = env.get_obs(camera_id=1) # wrist camera
    env.debug_save_obs(obs_head, 'data/obs_head')
    env.debug_save_obs(obs_wrist, 'data/obs_wrist')
    
    print(env.get_driller_pose())
        
    input("Press Enter to start the simulation...")
    
    # --------------------------------------step 1: move quadruped to dropping position--------------------------------------
    if not DISABLE_MOVE:
        forward_steps = 10000 # number of steps that quadruped walk to dropping position
        steps_per_camera_shot = 2 # number of steps per camera shot, increase this to reduce the frequency of camera shots and speed up the simulation
        head_camera_matrix = env.sim.humanoid_robot_cfg.camera_cfg[0].intrinsics
        head_camera_params = (head_camera_matrix[0, 0],head_camera_matrix[1, 1],head_camera_matrix[0, 2],head_camera_matrix[1, 2])

        # implement this to guide the quadruped to target dropping position
        #
        target_container_pose = obs_head.camera_pose
        target_container_pose[0, 3] += 0.5
        target_container_pose[1, 3] += 0.08
                
        def is_close(pose1, pose2, threshold): 
            R1, t1 = pose1[:3, :3], pose1[:3, 3]
            R2, t2 = pose2[:3, :3], pose2[:3, 3]

            trans_error = np.linalg.norm(t1[:2] - t2[:2])
            
            # print(f"Trans: {trans_error}")
            
            return trans_error < threshold, trans_error
        
        last_error = 1e6
        init_pose_container_world = []
        
        for step in range(forward_steps):
            if step % steps_per_camera_shot == 0:
                obs_head = env.get_obs(camera_id=0) # head camera
                trans_marker_world, rot_marker_world = detect_marker_pose(
                    detector, 
                    obs_head.rgb, 
                    head_camera_params,
                    obs_head.camera_pose,
                    tag_size=0.12
                )
                if trans_marker_world is not None:
                    # the container's pose is given by follows:
                    trans_container_world = rot_marker_world @ np.array([0,0.31,0.02]) + trans_marker_world
                    rot_container_world = rot_marker_world
                    pose_container_world = to_pose(trans_container_world, rot_container_world)
                    pose_container_world[0, 3] -= 0.6 if get_yaw_from_pose(pose_container_world) > 0 else 0
                
                    # print("Target container pose:", target_container_pose[:3, 3])
                    # print("Detected Container pose:", pose_container_world[:3, 3])
                    # print("True Container Pose:", env.get_container_pose()[:3, 3])
            
            if step == 0:
                init_pose_container_world = pose_container_world.copy()
                init_yaw = get_yaw_from_pose(pose_container_world)
                target_yaw = -1 * init_yaw

            quad_command = forward_quad_policy(pose_container_world, target_container_pose, target_yaw)
            # print(quad_command)
            move_head = False
        
            flag, error = is_close(pose_container_world, target_container_pose, threshold=0.01)
            if last_error > 0.2 and error < 0.2:
                move_head = True
                
            last_error = error
            
            if move_head:
                # if you need moving head to track the marker, implement this
                head_qpos = [0, 0.6]
                env.step_env(
                    humanoid_head_qpos=head_qpos,
                    quad_command=quad_command
                )
            else:
                env.step_env(quad_command=quad_command)
            
            
            if flag: 
                quad_command = np.array([0,0,0])
                env.step_env(quad_command=quad_command)
                break
            
            if step == forward_steps - 1:
                quad_command = np.array([0,0,0])
                env.step_env(quad_command=quad_command)

    print("Quadruped moved to target position.")
    # --------------------------------------step 2: detect driller pose------------------------------------------------------
    
    pose_est_ckpt_path = "/home/iros/桌面/eai_hw4/pose_est/est_pose_PointNet_euler_geodesic_1e-3_64_generate_cut_06071059_adaptive_0.1/checkpoint/checkpoint_12500.pth"  
    pose_est_config = Pose_Est_Config.from_yaml(get_exp_config_from_checkpoint(pose_est_ckpt_path))
    pose_est_model = get_pose_est_model(pose_est_config)
    pose_est_checkpoint = torch.load(pose_est_ckpt_path, map_location="cpu")
    pose_est_model.load_state_dict(pose_est_checkpoint["model"])
    pose_est_model = pose_est_model.eval().to("cuda:0")
    
    if not DISABLE_GRASP:
        obs_wrist = env.get_obs(camera_id=1) # wrist camera
        rgb, depth, camera_pose = obs_wrist.rgb, obs_wrist.depth, obs_wrist.camera_pose
        wrist_camera_matrix = env.sim.humanoid_robot_cfg.camera_cfg[1].intrinsics
        driller_pose = detect_driller_pose(rgb, depth, wrist_camera_matrix, camera_pose, pose_est_model)
        print(f"Driller Pose:\n{driller_pose}")
        driller_pose_gt = env.get_driller_pose()
        print(f"Ground Truth Driller Pose:\n{driller_pose_gt}")
        # metric judgement
        Metric['obj_pose'] = env.metric_obj_pose(driller_pose)
        
    # exit()


    # --------------------------------------step 3: plan grasp and lift------------------------------------------------------
    if not DISABLE_GRASP:
        obj_pose = driller_pose.copy()
        # print(obj_pose)
        grasps = get_grasps(args.obj) 
        grasps0_n = Grasp(grasps[0].trans, grasps[0].rot @ np.diag([-1,-1,1]), grasps[0].width)
        grasps2_n = Grasp(grasps[2].trans, grasps[2].rot @ np.diag([-1,-1,1]), grasps[2].width)
        valid_grasps = grasps # we have provided some grasps, you can choose to use them or yours
        grasp_config = dict( 
            reach_steps=50,
            lift_steps=50,
            delta_dist=0.002, 
        ) # the grasping design in assignment 2, you can choose to use it or design yours

        for obj_frame_grasp in valid_grasps:
            trans=obj_pose[:3, :3] @ obj_frame_grasp.trans + obj_pose[:3, 3]
            rot=obj_pose[:3, :3] @ obj_frame_grasp.rot
            rot=driller_pose_gt[:3, :3] @ obj_frame_grasp.rot
            rot_x = -1 if rot[0, 1] < 0 else 1
            rot_y = -1 if rot[1, 2] < 0 else 1
            rot_z = -1 if rot[2, 0] < 0 else 1
            R_in = np.array([[0, rot_x, 0], [0, 0 , rot_y], [rot_z, 0, 0]])
            R_target = np.array([
                [ 0, -1,  0],
                [ 0,  0,  1],
                [-1,  0,  0]
            ])
            M = R_in.T @ R_target
            rot = rot @ M
            
            trans[1] += 0.005
            
            robot_frame_grasp = Grasp(
                trans=trans,
                rot=rot,
                width=obj_frame_grasp.width,
            )
            grasp_plan = plan_grasp(env, robot_frame_grasp, grasp_config)
            if grasp_plan is not None:
                break
        if grasp_plan is None:
            print("No valid grasp plan found.")
            env.close()
            return
        reach_plan, lift_plan = grasp_plan
        

        pregrasp_plan = plan_move_qpos(observing_qpos, reach_plan[0], steps=100) # pregrasp, change if you want
        execute_plan(env, pregrasp_plan)
        open_gripper(env)
        execute_plan(env, reach_plan)
        close_gripper(env)
        execute_plan(env, lift_plan)


    # --------------------------------------step 4: plan to move and drop----------------------------------------------------
    if not DISABLE_GRASP and not DISABLE_MOVE:
        # implement your moving plan
        #
        final_lift_qpos = lift_plan[-1]

        begin_qpos = final_lift_qpos
        begin_trans, begin_rot = env.humanoid_robot_model.fk_eef(begin_qpos)

        end_trans = pose_container_world[:3, 3].copy()
        end_trans[2] += 0.25
        end_trans[1] -= 0.07
        # print(f"True Trans: {env.get_container_pose()[:3, 3]}")
        # print(f"End Trans: {end_trans}")
        # print(f"Begin Trans: {begin_trans}")

        # 高点
        middle_trans = (end_trans + begin_trans) / 2
        middle_trans[2] += 0.3
        middle_rot = np.array([
                [ 0, -1,  0],
                [ 0,  0,  1],
                [-1,  0,  0]
            ])
        
        trans_1 = (middle_trans + begin_trans) / 2
        rot_1 = middle_rot.copy()  # 姿态不变
        trans_2 = (middle_trans + end_trans) / 2
        rot_2 = middle_rot.copy()  # 姿态不变
        
        end_rot = middle_rot.copy()

        # 第一段：从抓完处 → 高点
        plan1 = plan_move(env, begin_qpos, begin_trans, begin_rot, trans_1, rot_1, steps=50)
        if plan1 is None:
            print("Plan to 1 failed")
            return
        
        execute_plan(env, plan1)
        
        qpos_1 = plan1[-1]
        plan_middle = plan_move(env, qpos_1, trans_1, rot_1, middle_trans, middle_rot, steps=50)
        if plan_middle is None:
            print("Plan to middle failed")
            return
        
        execute_plan(env, plan_middle)
        time.sleep(0.1)

        # 第二段：从高点 → container 上方
        qpos_middle = plan_middle[-1]
        plan2 = plan_move(env, qpos_middle, middle_trans, middle_rot, trans_2, rot_2, steps=100)
        if plan2 is None:
            print("Plan to 2 failed")
            return
        
        execute_plan(env, plan2)
        time.sleep(0.1)
        
        qpos_2 = plan2[-1]
        plan_end = plan_move(env, qpos_2, trans_2, rot_2, end_trans, end_rot, steps=100)
        if plan_end is None:
            print("Plan to container failed")
            return
        
        execute_plan(env, plan_end)
        time.sleep(0.5)
        open_gripper(env)


    # --------------------------------------step 5: move quadruped backward to initial position------------------------------
    if not DISABLE_MOVE:
        # implement
        #
        backward_steps = 10000 # customize by yourselves
        for step in range(backward_steps):
            # same as before, please implement this
            #
            
            if step % steps_per_camera_shot == 0:
                obs_head = env.get_obs(camera_id=0) # head camera
                trans_marker_world, rot_marker_world = detect_marker_pose(
                    detector, 
                    obs_head.rgb, 
                    head_camera_params,
                    obs_head.camera_pose,
                    tag_size=0.12
                )
                if trans_marker_world is not None:
                    # the container's pose is given by follows:
                    trans_container_world = rot_marker_world @ np.array([0,0.31,0.02]) + trans_marker_world
                    rot_container_world = rot_marker_world
                    pose_container_world = to_pose(trans_container_world, rot_container_world)
                    pose_container_world[0, 3] -= 0.6 if get_yaw_from_pose(pose_container_world) > 0 else 0
                
                    # print("Container pose:", pose_container_world)
                    # print("Target container pose:", init_pose_container_world[:3, 3])
                    # print("Detected Container pose:", pose_container_world[:3, 3])
                    # print("True Container Pose:", env.get_container_pose()[:3, 3])

            quad_command = backward_quad_policy(pose_container_world, init_pose_container_world, init_yaw)
            move_head = False
        
            flag, error = is_close(pose_container_world, init_pose_container_world, threshold=0.01)
            if last_error < 0.1 and error > 0.1:
                move_head = True
                
            last_error = error
            
            if move_head:
                # if you need moving head to track the marker, implement this
                head_qpos = [0,0.4]
                env.step_env(
                    humanoid_head_qpos=head_qpos,
                    quad_command=quad_command
                )
            else:
                env.step_env(quad_command=quad_command)
            
            
            if flag: break
        

    # test the metrics
    Metric["drop_precision"] = Metric["drop_precision"] or env.metric_drop_precision()
    Metric["quad_return"] = Metric["quad_return"] or env.metric_quad_return()

    print("Metrics:", Metric) 

    print("Simulation completed.")
    env.close()

if __name__ == "__main__":
    main()
