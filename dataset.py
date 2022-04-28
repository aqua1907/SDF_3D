import os
import numpy as np
import trimesh
import torch
import copy

from torch.utils.data.dataset import Dataset
from mesh_to_sdf import sample_sdf_near_surface, scale_to_unit_sphere
from mesh_to_sdf.surface_point_cloud import create_from_scans


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


class PointCloud(Dataset):
    def __init__(self, mesh_path, on_surface_points, keep_aspect_ratio=True):
        super().__init__()

        mesh = trimesh.load(mesh_path)
        mesh = scale_to_unit_sphere(mesh)
        surface_pc = create_from_scans(mesh, scan_count=100, scan_resolution=100)
        coords = surface_pc.points
        self.normals = surface_pc.normals

        # Reshape point cloud such that it lies in bounding box of (-1, 1) (distorts geometry, but makes for high
        # sample efficiency)
        coords -= np.mean(coords, axis=0, keepdims=True)
        if keep_aspect_ratio:
            coord_max = np.amax(coords)
            coord_min = np.amin(coords)
        else:
            coord_max = np.amax(coords, axis=0, keepdims=True)
            coord_min = np.amin(coords, axis=0, keepdims=True)

        self.coords = (coords - coord_min) / (coord_max - coord_min)
        self.coords -= 0.5
        self.coords *= 2.

        self.on_surface_points = on_surface_points

    def __len__(self):
        return self.coords.shape[0] // self.on_surface_points

    def __getitem__(self, idx):
        point_cloud_size = self.coords.shape[0]

        off_surface_samples = self.on_surface_points  # **2
        total_samples = self.on_surface_points + off_surface_samples

        # Random coords
        rand_idcs = np.random.choice(point_cloud_size, size=self.on_surface_points)

        on_surface_coords = self.coords[rand_idcs, :]
        on_surface_normals = self.normals[rand_idcs, :]

        off_surface_coords = np.random.uniform(-1, 1, size=(off_surface_samples, 3))
        off_surface_normals = np.ones((off_surface_samples, 3)) * -1

        sdf = np.zeros((total_samples, 1))  # on-surface = 0
        sdf[self.on_surface_points:, :] = -1  # off-surface = -1

        coords = np.concatenate((on_surface_coords, off_surface_coords), axis=0)
        normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)

        return torch.from_numpy(coords).float(), torch.from_numpy(sdf).float(), torch.from_numpy(normals).float()


if __name__ == "__main__":
    dataset = SDFDataset(r'objects/sword.obj')
    points, sdf_target = dataset[0]
    print(points.shape)
    print(type(points))
    print(sdf_target)
