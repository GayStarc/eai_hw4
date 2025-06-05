from dataclasses import dataclass
import numpy as np
from src.utils import to_pose
from transforms3d.quaternions import quat2mat

@dataclass
class TestData:
    table_trans: np.ndarray
    table_size: np.ndarray
    obj_trans: np.ndarray
    obj_quat: np.ndarray
    quad_reset_pos: np.ndarray

EXAMPLE_TEST_DATA = [
    TestData(
        table_trans = np.array([0.6, 0.38, 0.72]),
        table_size = np.array([0.68, 0.36, 0.02]),
        obj_trans = np.array([0.5, 0.3, 0.82]),
        obj_quat = np.array([1.0, 0.0, 0.0, 0.0]),
        quad_reset_pos = np.array([2.0, -0.2, 0.278])
    ),

    TestData(
        table_trans = np.array([0.6, 0.45, 0.72]),
        table_size = np.array([0.72, 0.42, 0.02]),
        obj_trans = np.array([0.53, 0.35, 0.82]),
        obj_quat = np.array([0.966, 0.0, 0.0, 0.259]),
        quad_reset_pos = np.array([1.8, -0.25, 0.278])
    ),

    TestData(
        table_trans = np.array([0.55, 0.45, 0.68]),
        table_size = np.array([0.68, 0.36, 0.02]),
        obj_trans = np.array([0.5, 0.4, 0.82]),
        obj_quat = np.array([0.924, 0.0, 0.0, -0.383]),
        quad_reset_pos = np.array([1.9, -0.15, 0.278])
    ),

    TestData(
        table_trans = np.array([0.6, 0.47, 0.74]),
        table_size = np.array([0.68, 0.36, 0.02]),
        obj_trans = np.array([0.48, 0.43, 0.82]),
        obj_quat = np.array([0.174, 0.0, 0.0, 0.985]),
        quad_reset_pos = np.array([1.7, -0.1, 0.278])
    ),
]

def load_test_data(id = 0):
    testdata = EXAMPLE_TEST_DATA[id]
    
    return dict(
        table_pose = to_pose(trans=testdata.table_trans),
        table_size = testdata.table_size,
        obj_pose = to_pose(testdata.obj_trans, quat2mat(testdata.obj_quat)),
        quad_reset_pos = testdata.quad_reset_pos
    )
    
def random_unit_quaternion():
    # 均匀采样单位四元数
    u1, u2, u3 = np.random.rand(3)
    q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    return np.array([q4, q1, q2, q3])  # w, x, y, z 格式

def generate_train_data(n=5000):
    dataset = []
    for _ in range(n):
        table_trans = np.random.uniform([0.55, 0.38, 0.68], [0.6, 0.47, 0.74])
        table_size = np.random.uniform([0.68, 0.36, 0.02], [0.72, 0.42, 0.02])  # z轴为常量
        obj_trans = np.random.uniform([0.48, 0.3, 0.82], [0.53, 0.43, 0.82])   # z轴为常量
        obj_quat = random_unit_quaternion()
        quad_reset_pos = np.random.uniform([1.7, -0.25, 0.278], [2.0, -0.1, 0.278])  # z轴为常量
        dataset.append(dict(
            table_pose = to_pose(trans=table_trans),
            table_size = table_size,
            obj_pose = to_pose(obj_trans, quat2mat(obj_quat)),
            quad_reset_pos = quad_reset_pos
        ))
    return dataset