import argparse
import math

import numpy as np

import mcubes

import torch

import trimesh

import models_class_cond, models_ae

from pathlib import Path


if __name__ == "__main__":

    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--ae', default='kl_d512_m512_l8', type=str, required=False) # 'kl_d512_m512_l16'
    parser.add_argument('--ae-pth', default='output/ae/kl_d512_m512_l8/checkpoint-199.pth', type=str, required=False) # 'output/ae/kl_d512_m512_l16/checkpoint-199.pth'
    parser.add_argument('--dm', default='kl_d512_m512_l8_d24_edm', type=str, required=False) # 'kl_d512_m512_l16_edm'
    parser.add_argument('--dm-pth', default='output/class_cond_dm/kl_d512_m512_l8_d24_edm/checkpoint-499.pth', type=str, required=False) # 'output/uncond_dm/kl_d512_m512_l16_edm/checkpoint-999.pth'
    parser.add_argument('--out-dir', default='class_cond_obj', type=str, required=False)
    args = parser.parse_args()
    print(args)

    Path("class_cond_obj/{}".format(args.dm)).mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda:0')

    ae = models_ae.__dict__[args.ae]()
    ae.eval()
    ae.load_state_dict(torch.load(args.ae_pth, weights_only=False)['model'])
    ae.to(device)

    model = models_class_cond.__dict__[args.dm]()
    model.eval()

    model.load_state_dict(torch.load(args.dm_pth, weights_only=False)['model'])
    model.to(device)

    density = 128
    gap = 2. / density
    x = np.linspace(-1, 1, density+1)
    y = np.linspace(-1, 1, density+1)
    z = np.linspace(-1, 1, density+1)
    xv, yv, zv = np.meshgrid(x, y, z)
    grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32)).view(3, -1).transpose(0, 1)[None].to(device, non_blocking=True)

    total = 1000
    iters = 100


    with torch.no_grad():
        for category_id in [18]:
            print(category_id)
            for i in range(1000//iters):
                sampled_array = model.sample(cond=torch.Tensor([category_id]*iters).long().to(device), batch_seeds=torch.arange(i*iters, (i+1)*iters).to(device)).float()

                print(sampled_array.shape, sampled_array.max(), sampled_array.min(), sampled_array.mean(), sampled_array.std())

                for j in range(sampled_array.shape[0]):
                    
                    logits = ae.decode(sampled_array[j:j+1], grid)

                    logits = logits.detach()
                    
                    volume = logits.view(density+1, density+1, density+1).permute(1, 0, 2).cpu().numpy()
                    verts, faces = mcubes.marching_cubes(volume, 0)

                    verts *= gap
                    verts -= 1


                    # m = trimesh.Trimesh(verts, faces)
                    # m.export(f'{args.out_dir}/{args.dm}/{category_id:02d}-{i*iters+j:05d}.obj')

                    out_base = f"{args.out_dir}/{args.dm}/{category_id:02d}-{i*iters+j:05d}"

                    # export mesh
                    m = trimesh.Trimesh(verts, faces)
                    m.export(out_base + ".obj")

                    # export latent
                    latent = sampled_array[j:j+1].detach().cpu()
                    torch.save(
                        {
                            "latent": latent,
                            "ae": args.ae,
                            "dm": args.dm,
                            "category_id": category_id,
                        },
                        out_base + "_latent.pt"
                    )
