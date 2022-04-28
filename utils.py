import os
import torch
import numpy as np
import trimesh
import random
import torch.backends.cudnn as cudnn

from mesh_to_sdf import scale_to_unit_sphere, mesh_to_sdf, sample_sdf_near_surface
from mesh_to_sdf.surface_point_cloud import create_from_scans
from datetime import datetime
import torch.utils.benchmark as benchmark

import pyrender


def F1_metric(model, data, batch_size, device):
    points, gt_sdf_sign = data
    with torch.no_grad():
        points = torch.from_numpy(points).to(device).float()
        sdf_pred = [model(batch)[0].cpu().numpy() for batch in torch.chunk(points, batch_size)]
        sdf_pred = np.concatenate(sdf_pred)
        sdf_pred = np.squeeze(sdf_pred)
        pred_sdf_sign = np.sign(sdf_pred)

    tp = np.sum(np.logical_and(gt_sdf_sign > 0, pred_sdf_sign > 0))
    fp = np.sum(np.logical_and(gt_sdf_sign < 0, pred_sdf_sign > 0))
    fn = np.sum(np.logical_and(gt_sdf_sign > 0, pred_sdf_sign < 0))

    f1 = tp / (tp + (fp + fn) * 0.5)

    return f1


def load_checkpoint(path, model=None, optimizer=None, scheduler=None):
    data = torch.load(path)
    if model is not None:
        model.load_state_dict(data["model"])

    if optimizer is not None:
        optimizer.load_state_dict(data["optimizer"])

    if scheduler is not None:
        scheduler.load_state_dict(data["scheduler"])


def save_checkpoint(path, epoch, model, optimizer, scheduler):
    save_data = dict(epoch=epoch,
                     model=model.state_dict(),
                     optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict())

    torch.save(save_data, path)


def create_gt_data(obj_path, scan_resolution=1000):
    filename = str(obj_path).replace('.obj', '')
    filename = filename + "_gt"
    npy_path = filename + '.npy'

    if os.path.exists(npy_path):
        data = np.load(npy_path, allow_pickle=True)
        points = data.item().get('points')
        gt_sdf_sign = data.item().get('gt_sdf_sign')
    else:
        mesh = trimesh.load(obj_path)
        mesh = scale_to_unit_sphere(mesh)
        surface_pc = create_from_scans(mesh, scan_count=100, scan_resolution=scan_resolution)
        points = surface_pc.points
        gt_sdf_sign = np.sign(mesh_to_sdf(mesh, points))

        np.save(npy_path, dict(points=points,
                               gt_sdf_sign=gt_sdf_sign))

    return points, gt_sdf_sign


def human_format(num):
    """
    Convert number to human readable format
    :param (float) num: number
    :return (string): string number
    """
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0

    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)


def init_torch_seeds(seed=0):
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    if seed == 0:  # slower, more reproducible
        cudnn.benchmark, cudnn.deterministic = False, True
    else:  # faster, less reproducible
        cudnn.benchmark, cudnn.deterministic = True, False


def create_run_folder(path):
    """
    Create folder for saving Tnesorboard events in the deep_homography net
    :param (string) path: Path to the folder
    :return (string): string of the name of folder
    """
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    folder = "run_" + dt_string
    run_folder_path = os.path.join(path, folder)
    os.mkdir(run_folder_path)  # create run folder
    os.mkdir(os.path.join(run_folder_path, "weights"))  # create weights folder
    print("[INFO] Run folder created")

    return folder


def model_forward(model, x):
    return model(x)


def model_benchmark(opt, model, data, device):
    model_params = sum(p.numel() for p in model.parameters())
    model_params = human_format(model_params)
    print(f"Number of parameters = {model_params}")
    print(f"Batch size = {opt['chunk_size']}")

    points = data[0][opt['chunk_size']]
    points = torch.from_numpy(points).to(device).float().to(device)
    t_model = benchmark.Timer(
        stmt='model_forward(model, x)',
        setup='from utils import model_forward',
        globals={'model': model, 'x': points}
    )

    m = t_model.blocked_autorange()
    print("Mean batch time: {:6.2f} ms".format(m.mean * 1e3))
    print("Mean sample time: {:6.2f} us".format(m.mean * 1e6 / opt['chunk_size']))


if __name__ == '__main__':
    pass
