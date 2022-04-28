import torch
import torch.nn.functional as F


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]

    return grad


def siren_loss_sdf(points, sdf_pred, sdf_target, gt_normals):
    """
       x: batch of input coordinates
       y: usually the output of the trial_soln function
    """

    grad = gradient(sdf_pred, points)

    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    sdf_constraint = torch.where(sdf_target != -1, sdf_pred, torch.zeros_like(sdf_pred))
    inter_constraint = torch.where(sdf_target != -1, torch.zeros_like(sdf_pred), torch.exp(-1e2 * torch.abs(sdf_pred)))
    normal_constraint = torch.where(sdf_target != -1, 1 - F.cosine_similarity(grad, gt_normals, dim=-1)[..., None],
                                    torch.zeros_like(grad[..., :1]))
    grad_constraint = torch.abs(grad.norm(dim=-1) - 1)

    loss = torch.abs(sdf_constraint).mean() * 3e3 + inter_constraint.mean() * 1e2 \
           + normal_constraint.mean() * 1e2 + grad_constraint.mean() * 5e1

    return loss
