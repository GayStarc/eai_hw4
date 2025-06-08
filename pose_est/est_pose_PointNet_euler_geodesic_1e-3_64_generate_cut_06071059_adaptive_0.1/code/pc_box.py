import os
import cv2
import numpy as np
from tqdm import tqdm
from scipy.stats import mode

def get_pc(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """ 你的原始点云生成函数 """
    height, width = depth.shape
    v, u = np.meshgrid(range(height), range(width), indexing="ij")
    u = u.flatten()
    v = v.flatten()
    depth_flat = depth.flatten()
    valid = depth_flat > 0
    pixels = np.stack([u[valid], v[valid], np.ones_like(u[valid])], axis=0)
    rays = np.linalg.inv(intrinsics) @ pixels
    return (rays * depth_flat[valid]).T

def compute_clipping_params(dataset_root: str, 
                          intrinsics: np.ndarray,
                          percentile: float = 0.0) -> dict:
    """
    统计原始点云分布特征（不进行任何过滤）
    
    Args:
        dataset_root: 数据集路径
        intrinsics: 相机内参矩阵 (3,3)
        percentile: 用于剔除离群点的百分位（建议0.02-0.05）
    
    Returns:
        {
            "table_height": 检测到的桌面高度,
            "pc_min": 建议的PC_MIN,
            "pc_max": 建议的PC_MAX,
            "obj_init_trans": 物体位置中位数,
            "obj_rand_range": 物体分布范围
        }
    """
    # 收集所有有效样本
    sample_dirs = [
        os.path.join(dataset_root, f) 
        for f in os.listdir(dataset_root) 
        if os.path.isdir(os.path.join(dataset_root, f))
    ]
    
    # 存储所有物体位姿和点云
    all_obj_poses = []
    all_points = []
    
    for dir_path in tqdm(sample_dirs, desc="Processing samples"):
        try:
            # 加载原始数据
            obj_pose = np.load(os.path.join(dir_path, "object_pose.npy"))
            camera_pose = np.load(os.path.join(dir_path, "camera_pose.npy"))
            depth = cv2.imread(os.path.join(dir_path, "depth.png"), cv2.IMREAD_UNCHANGED)
            
            # 生成点云并转换到世界坐标系
            pc_camera = get_pc(depth, intrinsics)
            pc_world = (camera_pose[:3, :3] @ pc_camera.T).T + camera_pose[:3, 3]
            
            all_points.append(pc_world)
            all_obj_poses.append(obj_pose[:3, 3])  # 只保存平移部分
        except:
            continue
    
    if not all_points:
        raise ValueError("No valid data found!")
    
    # 合并所有点云和位姿
    all_points = np.concatenate(all_points)
    all_obj_poses = np.array(all_obj_poses)
    
    z_values = all_points[:, 2].round(decimals=3)
    # 1. 检测桌面高度（z方向的众数）
    mode_result = mode(z_values.round(decimals=3))
    table_height = mode_result.mode if hasattr(mode_result, 'mode') else mode_result[0]
    table_height = float(table_height)
    
    obj_mask = all_points[:, 2] > table_height + 0.01
    if np.sum(obj_mask) == 0:
        raise ValueError("No object points found above table!")
    # 2. 计算物体点云分布（去掉桌面附近点）
    obj_points = all_points[all_points[:, 2] > table_height + 0.01]  # 略高于桌面
    
    # 3. 计算裁剪参数（使用百分位数剔除离群点）
    def get_percentile_bounds(data, axis=0):
        lower = np.percentile(data, percentile*100, axis=axis)
        upper = np.percentile(data, (1-percentile)*100, axis=axis)
        return lower, upper
    
    # 物体点云范围
    x_min, x_max = get_percentile_bounds(obj_points[:, 0])
    y_min, y_max = get_percentile_bounds(obj_points[:, 1])
    z_min, z_max = get_percentile_bounds(obj_points[:, 2])
    
    # 物体位置统计（使用中位数更鲁棒）
    obj_median_pos = np.median(all_obj_poses, axis=0)
    obj_pos_range = np.ptp(all_obj_poses[:, :2], axis=0)  # 仅x,y方向
    
    return {
        "table_height": float(table_height),
        "pc_min": [float(x_min), float(y_min), float(table_height + 0.005)],
        "pc_max": [float(x_max), float(y_max), float(z_max)],
        "obj_init_trans": obj_median_pos.tolist(),
        "obj_rand_range": float(np.max(obj_pos_range))  # 取x,y较大值
    }

if __name__ == "__main__":
    # 配置参数（需要根据实际修改）
    DATASET_PATH = "/media/liuzhuoyang/pku_course/eai/as2/data/power_drill/train"
    # INTRINSICS = np.array([
    #     [600, 0, 320],
    #     [0, 600, 240],
    #     [0, 0, 1]
    # ])  # 示例内参，需要替换为真实值
    INTRINSICS=np.array([[326, 0, 320], [0, 326, 180], [0, 0, 1]])
    
    try:
        params = compute_clipping_params(DATASET_PATH, INTRINSICS)
        
        print("\n自动计算的裁剪参数：")
        print(f"TABLE_HEIGHT = {params['table_height']:.3f}")
        print(f"OBJ_INIT_TRANS = np.array({params['obj_init_trans']})")
        print(f"OBJ_RAND_RANGE = {params['obj_rand_range']:.3f}")
        print(f"PC_MIN = np.array({params['pc_min']})  # 略高于桌面")
        print(f"PC_MAX = np.array({params['pc_max']})")
        
        print("\n建议直接替换原常量：")
        print(f"TABLE_HEIGHT = {params['table_height']:.3f}")
        print(f"OBJ_INIT_TRANS = np.array([{params['obj_init_trans'][0]:.3f}, "
              f"{params['obj_init_trans'][1]:.3f}, {params['obj_init_trans'][2]:.3f}])")
        print(f"OBJ_RAND_RANGE = {params['obj_rand_range']:.3f}")
        print(f"PC_MIN = np.array([{params['pc_min'][0]:.3f}, {params['pc_min'][1]:.3f}, "
              f"{params['pc_min'][2]:.3f}])")
        print(f"PC_MAX = np.array([{params['pc_max'][0]:.3f}, {params['pc_max'][1]:.3f}, "
              f"{params['pc_max'][2]:.3f}])")
    except Exception as e:
        print(f"Error: {str(e)}")