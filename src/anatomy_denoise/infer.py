import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from anatomy_denoise.models.stage1_teacher import Stage1Teacher
from anatomy_denoise.models.stage2_denoiser import Stage2Denoiser


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ldct_npy", type=Path, required=True)
    parser.add_argument("--stage1_ckpt", type=Path, required=True)
    parser.add_argument("--stage2_ckpt", type=Path, required=True)
    parser.add_argument("--out_npy", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ldct = np.load(args.ldct_npy).astype(np.float32)
    x_ldct = torch.from_numpy(ldct).unsqueeze(0).unsqueeze(0).to(device)

    stage1 = Stage1Teacher().to(device)
    stage1.load_state_dict(torch.load(args.stage1_ckpt, map_location=device)["model"])
    stage1.eval()
    for p in stage1.parameters():
        p.requires_grad = False

    stage2 = Stage2Denoiser().to(device)
    stage2.load_state_dict(torch.load(args.stage2_ckpt, map_location=device)["model"])
    stage2.eval()

    with torch.no_grad():
        t1 = stage1(x_ldct)
        s = t1["S"]
        e_a = t1["e_a"]
        s_scales = [
            F.interpolate(s, size=(256, 256), mode="bilinear", align_corners=False),
            F.interpolate(s, size=(128, 128), mode="bilinear", align_corners=False),
            F.interpolate(s, size=(64, 64), mode="bilinear", align_corners=False),
            F.interpolate(s, size=(32, 32), mode="bilinear", align_corners=False),
        ]
        # DDIM-like simplified inference over residual noise.
        z = torch.randn_like(x_ldct)
        for i in reversed(range(args.steps)):
            t = torch.full((1,), i * (1000 // args.steps), device=device, dtype=torch.long)
            pred = stage2(z, x_ldct, t, s_scales, e_a)["pred_residual"]
            z = z - pred / float(args.steps)
        denoised = (x_ldct + z).squeeze().cpu().numpy()

    args.out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_npy, denoised.astype(np.float32))
    print(f"Saved denoised slice to: {args.out_npy}")


if __name__ == "__main__":
    main()

