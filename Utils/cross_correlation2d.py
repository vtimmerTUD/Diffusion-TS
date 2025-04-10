import torch
from torch import nn

def cacf_torch(x, max_lag):
    def get_lower_triangular_indices(n):
        return [list(x) for x in torch.tril_indices(n, n)]

    # Normalize across rows (dim=0)
    x = (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + 1e-6)

    ind = get_lower_triangular_indices(x.shape[1])  # x.shape[1] is C

    x_l = x[:, ind[0]]  # Shape: [N, num_pairs]
    x_r = x[:, ind[1]]  # Shape: [N, num_pairs]

    cacf_list = []
    for i in range(max_lag):
        if i > 0:
            y = x_l[i:, :] * x_r[:-i, :]
        else:
            y = x_l * x_r
        cacf_i = torch.mean(y, dim=0, keepdim=True)  # Shape: [1, num_pairs]
        cacf_list.append(cacf_i)

    cacf = torch.cat(cacf_list, dim=0)  # Shape: [max_lag, num_pairs]
    return cacf.unsqueeze(0)  # Shape: [1, max_lag, num_pairs] to match original usage

class Loss(nn.Module):
    def __init__(self, name, reg=1.0, transform=lambda x: x, threshold=10., backward=False, norm_foo=lambda x: x):
        super(Loss, self).__init__()
        self.name = name
        self.reg = reg
        self.transform = transform
        self.threshold = threshold
        self.backward = backward
        self.norm_foo = norm_foo

    def forward(self, x_fake):
        self.loss_componentwise = self.compute(x_fake)
        return self.reg * self.loss_componentwise.mean()

    def compute(self, x_fake):
        raise NotImplementedError()

    @property
    def success(self):
        return torch.all(self.loss_componentwise <= self.threshold)


class CrossCorrelLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(CrossCorrelLoss, self).__init__(norm_foo=lambda x: torch.abs(x).sum(0), **kwargs)
        self.cross_correl_real = cacf_torch(self.transform(x_real), 1).mean(0)[0]

    def compute(self, x_fake):
        cross_correl_fake = cacf_torch(self.transform(x_fake), 1).mean(0)[0]
        loss = self.norm_foo(cross_correl_fake - self.cross_correl_real.to(x_fake.device))
        return loss / 10.