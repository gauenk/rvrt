# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import torch as th
import argparse
import cv2
import glob
import os
import torch
import requests
import numpy as np
from os import path as osp
import pandas as pd
from collections import OrderedDict
from torch.utils.data import DataLoader

from dev_basics.trte import bench
# from models.network_rvrt import RVRT as net
from rvrt.original import net
from utils import utils_image as util
from main_test_rvrt import test_video
# from data.dataset_video_test import VideoRecurrentTestDataset, VideoTestVimeo90KDataset, SingleVideoRecurrentTestDataset


def get_model(args):
    ''' prepare model and dataset according to args.task. '''

    # define model
    if args.task == '001_RVRT_videosr_bi_REDS_30frames':
        model = net(upscale=4, clip_size=2, img_size=[2, 64, 64],
                    window_size=[2, 8, 8], num_blocks=[1, 2, 1],
                    depths=[2, 2, 2], embed_dims=[144, 144, 144], num_heads=[6, 6, 6],
                    inputconv_groups=[1, 1, 1, 1, 1, 1], deformable_groups=12,
                    attention_heads=12,
                    attention_window=[3, 3], cpu_cache_length=100)
        datasets = ['REDS4']
        args.scale = 4
        args.window_size = [2,8,8]
        args.nonblind_denoising = False

    elif args.task in ['002_RVRT_videosr_bi_Vimeo_14frames', '003_RVRT_videosr_bd_Vimeo_14frames']:
        model = net(upscale=4, clip_size=2, img_size=[2, 64, 64], window_size=[2, 8, 8], num_blocks=[1, 2, 1],
                    depths=[2, 2, 2], embed_dims=[144, 144, 144], num_heads=[6, 6, 6],
                    inputconv_groups=[1, 1, 1, 1, 1, 1], deformable_groups=12, attention_heads=12,
                    attention_window=[3, 3], cpu_cache_length=100)
        datasets = ['Vid4'] # 'Vimeo'. Vimeo dataset is too large. Please refer to #training to download it.
        args.scale = 4
        args.window_size = [2,8,8]
        args.nonblind_denoising = False

    elif args.task in ['004_RVRT_videodeblurring_DVD_16frames']:
        model = net(upscale=1, clip_size=2, img_size=[2, 64, 64], window_size=[2, 8, 8], num_blocks=[1, 2, 1],
                    depths=[2, 2, 2], embed_dims=[192, 192, 192], num_heads=[6, 6, 6],
                    inputconv_groups=[1, 3, 3, 3, 3, 3], deformable_groups=12, attention_heads=12,
                    attention_window=[3, 3], cpu_cache_length=100)
        datasets = ['DVD10']
        args.scale = 1
        args.window_size = [2,8,8]
        args.nonblind_denoising = False

    elif args.task in ['005_RVRT_videodeblurring_GoPro_16frames']:
        model = net(upscale=1, clip_size=2, img_size=[2, 64, 64],
                    window_size=[2, 8, 8], num_blocks=[1, 2, 1],
                    depths=[2, 2, 2], embed_dims=[192, 192, 192], num_heads=[6, 6, 6],
                    inputconv_groups=[1, 3, 3, 3, 3, 3], deformable_groups=12,
                    attention_heads=12,
                    attention_window=[3, 3], cpu_cache_length=100)
        datasets = ['GoPro11-part1', 'GoPro11-part2']
        args.scale = 1
        args.window_size = [2,8,8]
        args.nonblind_denoising = False


    elif args.task == '006_RVRT_videodenoising_DAVIS_16frames':
        model = net(upscale=1, clip_size=2, img_size=[-1, 2, 11],
                    window_size=[2, 8, 8], num_blocks=[1, 2, 1],
                    depths=[2, 2, 2], embed_dims=[192, 192, 192], num_heads=[6, 6, 6],
                    inputconv_groups=[1, 3, 4, 6, 8, 4], deformable_groups=12,
                    attention_heads=12, attention_window=[3, 3],
                    nonblind_denoising=True, cpu_cache_length=100)
        datasets = ['Set8', 'DAVIS-test']
        args.scale = 1
        args.window_size = [2,8,8]
        args.nonblind_denoising = True
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='001_RVRT_videosr_bi_REDS_30frames', help='tasks: 001 to 006')
    parser.add_argument('--sigma', type=int, default=0, help='noise level for denoising: 10, 20, 30, 40, 50')
    parser.add_argument('--folder_lq', type=str, default='testsets/REDS4/sharp_bicubic',
                        help='input low-quality test video folder')
    parser.add_argument('--folder_gt', type=str, default=None,
                        help='input ground-truth test video folder')
    parser.add_argument('--tile', type=int, nargs='+', default=[0,256,256],
                        help='Tile size, [0,0,0] for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, nargs='+', default=[2,20,20],
                        help='Overlapping of different tiles')
    parser.add_argument('--num_workers', type=int, default=16, help='number of workers in data loading')
    parser.add_argument('--save_result', action='store_true', help='save resulting image')
    args = parser.parse_args()

    # define model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -- benchmarking --
    tasks =  ['001_RVRT_videosr_bi_REDS_30frames',
              '002_RVRT_videosr_bi_Vimeo_14frames',
              '004_RVRT_videodeblurring_DVD_16frames',
              '005_RVRT_videodeblurring_GoPro_16frames',
              '006_RVRT_videodenoising_DAVIS_16frames']
    nchannels = [3,3,3,3,4,4]
    results = []
    for task,chnls in zip(tasks,nchannels):
        if not("deno" in task): continue

        # -- shape --
        if "videosr" in task:
            _task = "sr"
        elif "videodeblurring" in task:
            _task = "deblur"
        elif "videodenoising" in task:
            _task = "deno"
            # print(model)
        else:
            _task = "idk"
        if _task == "sr":
            vshape = (1,5,chnls,156,156)
        else:
            # vshape = (4,4,chnls,256,256)
            # vshape = (1,3,chnls,512,512)
            # vshape = (1,85,chnls,540,960)
            # vshape = (1,20,chnls,256,256)
            # vshape = (1,85,chnls,540,960)
            vshape = (2,8,chnls,156,156)
        if _task == "sr": continue

        # -- view --
        args.task = task
        model = get_model(args)
        model.eval()
        model = model.to(device)

        # print(model(

        if _task != "sr":
            vid = th.randn(vshape).to(device)
            out = model(vid)
            print(out.shape,vid.shape)

        # -- run summary --
        # res = bench.summary_loaded(model,vshape,with_flows=False)
        def fwd(lq,*_args,**kwargs):
            output = test_video(lq, model, args)
            return output

        res = bench.run_fwd_vshape(fwd,vshape,with_flows=False)
        res.task = _task
        results.append(res)

    results = pd.DataFrame(results)
    print(results)
    print(results.columns)
    print(results[['timer_fwd_nograd','trainable_params',"alloc_fwd_nograd"]])
    # print(results[['timer_fwd','timer_bwd','task']])
    # print(results[['alloc_fwd','alloc_bwd','res_fwd','res_bwd','task']])
    # print(results[['fwdbwd_mem','trainable_params','macs','task']])
    # results.to_csv("bench_results.csv",index=False)


if __name__ == '__main__':
    main()
