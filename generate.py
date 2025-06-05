import argparse
from typing import Optional, Tuple, List
import numpy as np
import cv2
from pyapriltags import Detector

from src.type import Grasp
from src.utils import to_pose
from src.sim.wrapper_env import WrapperEnvConfig, WrapperEnv
from src.sim.wrapper_env import get_grasps
from src.test.load_test import generate_train_data

from pose_est.utils import get_exp_config_from_checkpoint, farthest_point_sampling, get_pc
import torch
import os

def depth_to_pc(depth, intrinsics):
    h, w = depth.shape
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    u = np.arange(w)
    v = np.arange(h)
    u, v = np.meshgrid(u, v)
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pc = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    return pc

def detect_driller_pose(img, depth, camera_matrix, camera_pose, model, *args, **kwargs):
    """
    Detects the pose of driller, you can include your policy in args
    """
    # implement the detection logic here
    pc = get_pc(depth, camera_matrix) * np.array([-1, -1, 1])
    point_num = 1024  
    fps_indices = farthest_point_sampling(pc, point_num)
    pc_sampled = pc[fps_indices]
    pc_input = torch.tensor(pc_sampled, dtype=torch.float32).unsqueeze(0).to("cuda:0")  # (1, N, 3)
    trans_pred, rot_pred = model.est(pc_input) 
    obj_pose_in_camera = np.eye(4)
    obj_pose_in_camera[:3, :3] = rot_pred.detach().cpu().numpy().copy()
    obj_pose_in_camera[:3, 3] = trans_pred.detach().cpu().numpy().copy()
    obj_pose_in_world = camera_pose @ obj_pose_in_camera
    
    return obj_pose_in_world

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

def main():
    parser = argparse.ArgumentParser(description="Launcher config - Physics")
    parser.add_argument("--robot", type=str, default="galbot")
    parser.add_argument("--obj", type=str, default="power_drill")
    parser.add_argument("--ctrl_dt", type=float, default=0.02)
    parser.add_argument("--headless", type=int, default=0)
    parser.add_argument("--reset_wait_steps", type=int, default=100)

    args = parser.parse_args()

    env_config = WrapperEnvConfig(
        humanoid_robot=args.robot,
        obj_name=args.obj,
        headless=args.headless,
        ctrl_dt=args.ctrl_dt,
        reset_wait_steps=args.reset_wait_steps,
    )


    env = WrapperEnv(env_config)
    dataset_train = generate_train_data(5000)
    dataset_val = generate_train_data(500)
    
    train_root = f"/home/iros/桌面/eai_hw4/data_generated/power_drill/train"
    val_root = f"/home/iros/桌面/eai_hw4/data_generated/power_drill/val"
    
    for i, data_dict in enumerate(dataset_train):
            
        env.set_table_obj_config(
            table_pose=data_dict['table_pose'],
            table_size=data_dict['table_size'],
            obj_pose=data_dict['obj_pose']
        )
        env.set_quad_reset_pos(data_dict['quad_reset_pos'])

        env.launch()
        env.reset(humanoid_qpos=env.sim.humanoid_robot_cfg.joint_init_qpos)
        humanoid_init_qpos = env.sim.humanoid_robot_cfg.joint_init_qpos[:7]
        
        head_init_qpos = np.array([0.0,0.4])

        env.step_env(humanoid_head_qpos=head_init_qpos)
        
        observing_qpos = humanoid_init_qpos + np.array([-0.2,-0.1,0.3,0,0.1,0.3,0]) 
        init_plan = plan_move_qpos(humanoid_init_qpos, observing_qpos, steps = 50)
        execute_plan(env, init_plan)
        
        obs_wrist = env.get_obs(camera_id=1) 
        driller_pose = env.get_driller_pose()
        
        save_dir = os.path.join(train_root, f"{i}")
        env.debug_save_obs(obs_wrist, save_dir)
        np.save(os.path.join(save_dir, "object_pose.npy"), driller_pose)
        
        env.close()
        
    for i, data_dict in enumerate(dataset_val):
    
        env.set_table_obj_config(
            table_pose=data_dict['table_pose'],
            table_size=data_dict['table_size'],
            obj_pose=data_dict['obj_pose']
        )
        env.set_quad_reset_pos(data_dict['quad_reset_pos'])

        env.launch()
        env.reset(humanoid_qpos=env.sim.humanoid_robot_cfg.joint_init_qpos)
        humanoid_init_qpos = env.sim.humanoid_robot_cfg.joint_init_qpos[:7]
        
        head_init_qpos = np.array([0.0,0.4])

        env.step_env(humanoid_head_qpos=head_init_qpos)
        
        observing_qpos = humanoid_init_qpos + np.array([-0.2,-0.1,0.3,0,0.1,0.3,0]) 
        init_plan = plan_move_qpos(humanoid_init_qpos, observing_qpos, steps = 50)
        execute_plan(env, init_plan)
        
        obs_wrist = env.get_obs(camera_id=1) 
        driller_pose = env.get_driller_pose()
        
        save_dir = os.path.join(val_root, f"{i}")
        env.debug_save_obs(obs_wrist, save_dir)
        np.save(os.path.join(save_dir, "object_pose.npy"), driller_pose)
        
        env.close()

    print("Generation completed.")

if __name__ == "__main__":
    main()
