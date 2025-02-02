import os
import random
import copy
import torch
import numpy as np
import csv

from matplotlib import pyplot as plt

# Treering utils, reference: https://github.com/YuxinWenRick/tree-ring-watermark/tree/main

def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def circle_mask(height=17117, width=44, r=10, x_offset=0, y_offset=0):
    # reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
    x0 = width // 2
    y0 = height // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:height, :width]
    y = y[::-1]

    return ((x - x0)**2 + (y-y0)**2)<= r**2


def get_watermarking_mask(init_latents_w, device, w_mask_shape='circle'):
    watermarking_mask = torch.zeros(init_latents_w.shape, dtype=torch.bool).to(device)

    if w_mask_shape == 'circle':
        np_mask = circle_mask(height=init_latents_w.shape[-2], width=init_latents_w.shape[-1], r=args.w_radius)
        torch_mask = torch.tensor(np_mask).to(device)

        watermarking_mask[:, :] = torch_mask
        # if args.w_channel == -1:
        #     # all channels
        #     watermarking_mask[:, :] = torch_mask
        # else:
        #     watermarking_mask[:, args.w_channel] = torch_mask
    elif w_mask_shape == 'square':
        anchor_p = init_latents_w.shape[-1] // 2
        watermarking_mask[:, :, anchor_p-args.w_radius:anchor_p+args.w_radius, anchor_p-args.w_radius:anchor_p+args.w_radius] = True

        # if args.w_channel == -1:
        #     # all channels
        #     watermarking_mask[:, :, anchor_p-args.w_radius:anchor_p+args.w_radius, anchor_p-args.w_radius:anchor_p+args.w_radius] = True
        # else:
        #     watermarking_mask[:, args.w_channel, anchor_p-args.w_radius:anchor_p+args.w_radius, anchor_p-args.w_radius:anchor_p+args.w_radius] = True
    elif w_mask_shape == 'no':
        pass
    else:
        raise NotImplementedError(f'w_mask_shape: {args.w_mask_shape}')

    return watermarking_mask


def inject_watermark(init_latents_w, watermarking_mask, gt_patch, w_injection='complex'):
    init_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(init_latents_w), dim=(-1, -2))
    if w_injection == 'complex':
        init_latents_w_fft[watermarking_mask] = gt_patch[watermarking_mask].clone()
    elif w_injection == 'seed':
        init_latents_w[watermarking_mask] = gt_patch[watermarking_mask].clone()
        return init_latents_w
    else:
        NotImplementedError(f'w_injection: {w_injection}')

    init_latents_w = torch.fft.ifft2(torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))).real

    return init_latents_w


def get_watermarking_pattern(device, shape, seed, w_pattern='ring', w_radius=100):
    set_random_seed(seed)
    gt_init = torch.randn(shape, device=device)

    if 'seed_ring' in w_pattern:
        gt_patch = gt_init

        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-2], gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(device)
            
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
    elif 'seed_zeros' in w_pattern:
        gt_patch = gt_init * 0
    elif 'seed_rand' in w_pattern:
        gt_patch = gt_init
    elif 'rand' in w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch[:] = gt_patch[0]
    elif 'zeros' in w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
    elif 'const' in w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
        w_pattern_const = 0 # TODO: remove hardcore i guess
        gt_patch += w_pattern_const
    elif 'ring' in w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))

        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-2], gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(device)
            
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, i, 0].item()

    return gt_patch


def eval_watermark(reversed_latents, watermarking_mask, gt_patch, w_measurement='l1_complex'):
    reversed_latents = reversed_latents.unsqueeze(0) # think only one unsqueeze is necessary here
    print(w_measurement)
    if 'complex' in w_measurement:
        reversed_latents_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents), dim=(-1, -2))
        target_patch = gt_patch
    else:
        NotImplementedError(f'w_measurement: {w_measurement}')
    if 'l1' in w_measurement:
        metric = torch.abs(reversed_latents_fft[watermarking_mask] - target_patch[watermarking_mask]).mean().item()
    else:
        NotImplementedError(f'w_measurement: {w_measurement}')
    
    return metric
    