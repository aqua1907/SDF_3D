import os
import numpy as np
import trimesh
import torch
import copy

from torch.utils.data.dataset import Dataset
from mesh_to_sdf import sample_sdf_near_surface


class SDFDataset(Dataset):
    def __init__(self, obj_path, batch_size=512, num_points=250000, scan_resolution=1000,
                 clip_values=False, min_max_value=0.1):
        self.batch_size = batch_size
        self.clip_values = clip_values
        self.min_max = min_max_value

        npy_path = str(obj_path).replace('.obj', '.npy')
        if os.path.exists(npy_path):
            data = np.load(npy_path, allow_pickle=True)
            self.points = data.item().get('points')
            self.sdf_target = data.item().get('sdf_target')

            if len(self.sdf_target) != num_points:
                raise AssertionError(f"Amount of sdf_target not equals number of input points. "
                                     f"[sdf_target={len(self.sdf_target)}] != [num_points={num_points}]")
        else:
            mesh = trimesh.load(obj_path)
            points, sdf = sample_sdf_near_surface(mesh,
                                                  number_of_points=num_points,
                                                  scan_resolution=scan_resolution)
            indexes = np.arange(len(sdf))
            np.random.shuffle(indexes)

            self.points = points[indexes]
            self.sdf_target = sdf[indexes]

            np.save(npy_path,
                    dict(points=self.points,
                         sdf_target=self.sdf_target))

    def __len__(self):
        return len(self.sdf_target) // self.batch_size + 1

    def __getitem__(self, idx):
        st_idx = self.batch_size * idx
        end_idx = min(self.batch_size * (idx + 1), len(self.sdf_target))
        tensor_points = torch.from_numpy(self.points[st_idx: end_idx])
        tensor_sdf_target = torch.from_numpy(self.sdf_target[st_idx: end_idx])
        if self.clip_values:
            tensor_sdf_target = torch.clamp(tensor_sdf_target, -self.min_max, self.min_max)

        return tensor_points, tensor_sdf_target


if __name__ == "__main__":
    dataset = SDFDataset(r'objects/sword.obj')
    points, sdf_target = dataset[0]
    print(points.shape)
    print(type(points))
    print(sdf_target)
