from typing import Tuple, Dict
import torch
from torch import nn
import torch.nn.functional as F

from ..config import Config

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]

    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack((b1, b2, b3), dim=-2)

def geodesic_loss(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    R_diff = torch.matmul(R_pred.transpose(-1, -2), R_gt)
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(cos_theta)
    return theta.mean()

class EstPoseNet(nn.Module):

    config: Config

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.feat = nn.Sequential( 
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.trans_head = nn.Linear(256, 3) 
        self.rot6d_head = nn.Linear(256, 6) 
        
    def forward(
        self, pc: torch.Tensor, trans: torch.Tensor, rot: torch.Tensor, **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """
        Forward of EstPoseNet

        Parameters
        ----------
        pc : torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)
        trans : torch.Tensor
            Ground truth translation vector in camera frame, shape \(B, 3\)
        rot : torch.Tensor
            Ground truth rotation matrix in camera frame, shape \(B, 3, 3\)

        Returns
        -------
        float
            The loss value according to ground truth translation and rotation
        Dict[str, float]
            A dictionary containing additional metrics you want to log
        """
        pc = pc.transpose(1, 2)
        feat = self.feat(pc)  
        global_feat = torch.max(feat, 2)[0]  
        x = self.fc(global_feat) 

        pred_trans = self.trans_head(x)  
        pred_rot6d = self.rot6d_head(x)  
        pred_rot = rotation_6d_to_matrix(pred_rot6d)  

        loss_trans = F.mse_loss(pred_trans, trans)
        loss_rot = geodesic_loss(pred_rot, rot)

        loss = loss_trans + loss_rot

        metric = dict(
            loss=loss.item(),
            loss_trans=loss_trans.item(),
            loss_rot=loss_rot.item()
        )
        return loss, metric

    def est(self, pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate translation and rotation in the camera frame

        Parameters
        ----------
        pc : torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)

        Returns
        -------
        trans: torch.Tensor
            Estimated translation vector in camera frame, shape \(B, 3\)
        rot: torch.Tensor
            Estimated rotation matrix in camera frame, shape \(B, 3, 3\)

        Note
        ----
        The rotation matrix should satisfy the requirement of orthogonality and determinant 1.
        """
        pc = pc.transpose(1, 2)
        feat = self.feat(pc)  
        global_feat = torch.max(feat, 2)[0] 
        x = self.fc(global_feat) 

        pred_trans = self.trans_head(x)  
        pred_rot6d = self.rot6d_head(x)  
        pred_rot = rotation_6d_to_matrix(pred_rot6d) 

        return pred_trans, pred_rot
    