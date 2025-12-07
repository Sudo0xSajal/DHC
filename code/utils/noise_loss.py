import torch
import torch.nn as nn

class NoiseLosses(nn.Module):
    """
    Provides:
    - L_sup_noise: supervise noise with |p - y_onehot| on labeled
    - L_u_disagree: on unlabeled, match noise to |pA - pB|
    - L_noise_cons: encourage nA ~ nB on the same input
    """
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='mean')
        self.mse = nn.MSELoss(reduction='mean')

    def labeled_target(self, p, y):
        # p: [B,C,...], y: [B,1,...] → |p - onehot(y)|
        tgt = torch.zeros_like(p)
        tgt.scatter_(1, y.long(), 1)
        return (p - tgt).abs()

    def forward(self,
                pA_l=None, pB_l=None, y_l=None, nA_l=None, nB_l=None,
                pA_u=None, pB_u=None, nA_u=None, nB_u=None,
                w_l=1.0, w_u=1.0, w_cons=1.0):
        device = None
        for t in [pA_l, pA_u, nA_l, nA_u, pB_l, pB_u]:
            if t is not None:
                device = t.device; break
        zero = torch.tensor(0.0, device=device)
        L_sup, L_dis, L_cons = zero, zero, zero

        if (pA_l is not None) and (y_l is not None) and (nA_l is not None) and (nB_l is not None):
            tgt = self.labeled_target(pA_l, y_l)
            L_sup = self.l1(nA_l, tgt) + self.l1(nB_l, tgt)
            L_sup = w_l * L_sup

        if (pA_u is not None) and (pB_u is not None) and (nA_u is not None) and (nB_u is not None):
            disagree = (pA_u - pB_u).abs()
            L_dis = self.mse(nA_u, disagree) + self.mse(nB_u, disagree)
            L_dis = w_u * L_dis
            L_cons = self.mse(nA_u, nB_u) * w_cons

        return L_sup, L_dis, L_cons