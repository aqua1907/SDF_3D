import os.path
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

import utils
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import SDFDataset
from models.siren import SIREN

SEED = 123
utils.init_seeds(SEED)


def train(opt, obj_path, device, checkpoint, val_data, weights=None):
    # Create dataset and dataloader
    dataset = SDFDataset(obj_path, opt['chunk_size'],
                         opt['num_points'], opt['scan_resolution'],
                         opt['clip_values'], opt['min_max'])
    train_loader = DataLoader(dataset, batch_size=opt['batch_size'], shuffle=True, num_workers=0)

    # Initialise model
    model = SIREN(opt['dense_layers'],
                  opt['in_features'],
                  opt['out_features'],
                  opt['w0'], opt['w0_initial'],
                  initializer=opt['initializer'],
                  c=opt['c']).to(device)

    model_params = sum(p.numel() for p in model.parameters())
    model_params = utils.human_format(model_params)
    print(f"Number of parameters = {model_params}")

    if opt['loss'] == 'l1_loss':
        criterion = nn.L1Loss()
    elif opt['loss'] == 'l2_loss':
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), opt['init_lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=30, verbose=True,
                                                           threshold=0.0001, threshold_mode='abs')

    if weights is not None:
        utils.load_checkpoint(checkpoint, model, optimizer, scheduler)

    nb = len(train_loader)  # number of batches
    start_epoch = 0
    best_loss = np.inf
    scheduler.last_epoch = start_epoch - 1
    epochs = opt['epochs']

    for epoch in range(start_epoch, epochs):
        # Train
        model.train()
        losses = []
        avg_loss = 0.0

        loop = tqdm(enumerate(train_loader), total=nb)
        loop.set_description(f"Epoch [{epoch + 1}/{epochs}]")

        for i, data in loop:
            points, sdf_target = data
            points = points.to(device).squeeze(0)
            sdf_target = sdf_target.to(device).permute(1, 0)

            optimizer.zero_grad()

            sdf_pred = model(points)
            if opt['clip_values']:
                sdf_pred = torch.clamp(sdf_pred, -opt['min_max'], opt['min_max'])

            loss = criterion(sdf_pred, sdf_target)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            avg_loss = sum(losses) / len(losses)

            info = dict(train_loss=avg_loss)
            loop.set_postfix(info)

        if best_loss > avg_loss:
            best_loss = avg_loss
            utils.save_checkpoint(checkpoint / 'best.pth', epoch + 1, model, optimizer, scheduler)

        if (epoch + 1) % 150 == 0:
            model.eval()
            f1_value = utils.F1_metric(model, val_data, opt['chunk_size'], device)
            print(f"\nF1 = {f1_value:.3f}")

        scheduler.step(avg_loss)


if __name__ == "__main__":
    obj_path = Path(r'objects/sword.obj')
    obj_name = obj_path.name
    obj_name = obj_name.split('.')[0]
    ckpt_dir = Path(rf'checkpoints/{obj_name}')

    val_data = utils.create_gt_data(obj_path)

    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)

    # Hyperparameters
    with open(rf"configs/{obj_name}_config.yaml") as f:
        opt = yaml.load(f, Loader=yaml.SafeLoader)  # load hyperparameters and additional parameters
    f.close()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train(opt, obj_path, device, ckpt_dir, val_data)
