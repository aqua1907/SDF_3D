import torch
import utils
import os
import yaml
from pathlib import Path
from models.siren import SIREN


def evaluate(opt, checkpoint, data, device, benchmark=False):
    # Initialise model
    model = SIREN(opt['dense_layers'],
                  opt['in_features'],
                  opt['out_features'],
                  opt['w0'], opt['w0_initial'],
                  initializer=opt['initializer'],
                  c=opt['c']).to(device)
    utils.load_checkpoint(checkpoint, model)
    model.eval()

    f1_value = utils.F1_metric(model, data, opt['chunk_size'], device)
    print(f"F1 Score = {f1_value:.3f}")

    if benchmark:
        utils.model_benchmark(opt, model, data, device)


if __name__ == "__main__":
    objects = ['chair', 'handgun', 'pixar-lamp', 'plane', 'raven', 'spaceship', 'sword', 'wooden-coffee-table']
    for mesh in objects:
        print('-----------------------------------------------')
        print(f'{mesh}:')
        print('---')
        obj_path = Path(rf'objects/{mesh}.obj')
        print(f'3D object size = {obj_path.stat().st_size * 1e-6:.2f}MB')

        data = utils.create_gt_data(obj_path)
        obj_name = obj_path.name
        obj_name = obj_name.split('.')[0]
        ckpt_dir = rf'checkpoints/{obj_name}/best.pth'

        # Hyperparameters
        with open(rf"configs/{obj_name}_config.yaml") as f:
            opt = yaml.load(f, Loader=yaml.SafeLoader)  # load hyperparameters and additional parameters
        f.close()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        evaluate(opt, ckpt_dir, data, device, benchmark=True)
        print('-----------------------------------------------')
        print('\n')
