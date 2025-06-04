from typing import Tuple, Dict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ..config import Pose_Est_Config


class EstCoordNet(nn.Module):

    config: Pose_Est_Config

    def __init__(self, config: Pose_Est_Config):
        """
        Estimate the coordinates in the object frame for each object point.
        """
        super().__init__()
        self.config = config

        self.mlp1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Linear(64, 128), nn.ReLU())
        self.mlp3 = nn.Sequential(nn.Linear(128, 256), nn.ReLU())
        
        self.global_fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        self.decoder = nn.Sequential(
            nn.Linear(64 + 128 + 128, 128),  
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(
        self, pc: torch.Tensor, coord: torch.Tensor, **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """
        Forward of EstCoordNet

        Parameters
        ----------
        pc: torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)
        coord: torch.Tensor
            Ground truth coordinates in the object frame, shape \(B, N, 3\)

        Returns
        -------
        float
            The loss value according to ground truth coordinates
        Dict[str, float]
            A dictionary containing additional metrics you want to log
        """
        B, N, _ = pc.shape

        x1 = self.mlp1(pc)    
        x2 = self.mlp2(x1)   
        x3 = self.mlp3(x2)     

        x_max = torch.max(x3, dim=1, keepdim=True)[0]  
        x_global = self.global_fc(x_max)               
        x_global_expand = x_global.expand(-1, N, -1)  

        x_cat = torch.cat([x1, x2, x_global_expand], dim=-1)  

        pred_coord = self.decoder(x_cat) 

        loss = F.mse_loss(pred_coord, coord)

        return loss, {'loss_coord': loss.item()}
    
    def umeyama_alignment(self, src: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = src.shape

        mu_src = src.mean(dim=1, keepdim=True) 
        mu_tgt = tgt.mean(dim=1, keepdim=True) 

        src_centered = src - mu_src  
        tgt_centered = tgt - mu_tgt

        cov = torch.matmul(tgt_centered.transpose(1, 2), src_centered) / N 

        U, S, Vt = torch.linalg.svd(cov)

        d = torch.det(torch.matmul(U, Vt))
        D = torch.eye(3, device=src.device).unsqueeze(0).repeat(B, 1, 1)
        D[:, 2, 2] = torch.where(d < 0, -1.0, 1.0)

        R = torch.matmul(torch.matmul(U, D), Vt) 
        t = mu_tgt.squeeze(1) - torch.matmul(R, mu_src.squeeze(1).unsqueeze(-1)).squeeze(-1) 

        return R, t

    
    def ransac_fit(self, pc: torch.Tensor, pred_coord: torch.Tensor, max_iters=100, threshold=0.01):
        B, N, _ = pc.shape
        device = pc.device
        best_R = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
        best_t = torch.zeros(B, 3, device=device)
        best_inlier_count = torch.zeros(B, dtype=torch.long, device=device)

        for _ in range(max_iters):
            idx = torch.randint(0, N, (B, 3), device=device)  
            src = torch.gather(pred_coord, 1, idx.unsqueeze(-1).repeat(1, 1, 3))
            tgt = torch.gather(pc, 1, idx.unsqueeze(-1).repeat(1, 1, 3))

            R, t = self.umeyama_alignment(src, tgt)

            pred_pc = (pred_coord @ R.transpose(1, 2)) + t.unsqueeze(1)
            dist = torch.norm(pred_pc - pc, dim=-1) 
            inlier_mask = (dist < threshold)
            inlier_count = inlier_mask.sum(dim=-1)

            update_mask = inlier_count > best_inlier_count
            best_inlier_count = torch.where(update_mask, inlier_count, best_inlier_count)
            best_R = torch.where(update_mask.view(-1, 1, 1), R, best_R)
            best_t = torch.where(update_mask.view(-1, 1), t, best_t)

        return best_t, best_R
    
    def forward_coord_only(self, pc: torch.Tensor) -> torch.Tensor:
        B, N, _ = pc.shape

        x1 = self.mlp1(pc)    
        x2 = self.mlp2(x1)  
        x3 = self.mlp3(x2)   

        x_max = torch.max(x3, dim=1, keepdim=True)[0] 
        x_global = self.global_fc(x_max)              
        x_global_expand = x_global.expand(-1, N, -1) 

        x_cat = torch.cat([x1, x2, x_global_expand], dim=-1) 
        pred_coord = self.decoder(x_cat)                     

        return pred_coord

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

        We don't have a strict limit on the running time, so you can use for loops and numpy instead of batch processing and torch.

        The only requirement is that the input and output should be torch tensors on the same device and with the same dtype.
        """
        self.eval()
        with torch.no_grad():
            pred_coord = self.forward_coord_only(pc)  
            trans, rot = self.ransac_fit(pc, pred_coord)
        return trans, rot
