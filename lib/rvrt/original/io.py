
from easydict import EasyDict as edict
from .network_rvrt import RVRT as net
from ..utils.misc import optional

def load_model(cfg):
    task = edict(optional(cfg,"task","denoising"))
    model,datasets,args = init_from_task(task)
    return model

def get_model(task):
    ''' prepare model and dataset according to args.task. '''
    args = edict()
    args.task = task

    # define model
    if task == '001_RVRT_videosr_bi_REDS_30frames':
        model = net(upscale=4, clip_size=2, img_size=[2, 64, 64], window_size=[2, 8, 8], num_blocks=[1, 2, 1],
                    depths=[2, 2, 2], embed_dims=[144, 144, 144], num_heads=[6, 6, 6],
                    inputconv_groups=[1, 1, 1, 1, 1, 1], deformable_groups=12, attention_heads=12,
                    attention_window=[3, 3], cpu_cache_length=100)
        datasets = ['REDS4']
        args.scale = 4
        args.window_size = [2,8,8]
        args.nonblind_denoising = False

    elif task in ['002_RVRT_videosr_bi_Vimeo_14frames', '003_RVRT_videosr_bd_Vimeo_14frames']:
        model = net(upscale=4, clip_size=2, img_size=[2, 64, 64], window_size=[2, 8, 8], num_blocks=[1, 2, 1],
                    depths=[2, 2, 2], embed_dims=[144, 144, 144], num_heads=[6, 6, 6],
                    inputconv_groups=[1, 1, 1, 1, 1, 1], deformable_groups=12, attention_heads=12,
                    attention_window=[3, 3], cpu_cache_length=100)
        datasets = ['Vid4'] # 'Vimeo'. Vimeo dataset is too large. Please refer to #training to download it.
        args.scale = 4
        args.window_size = [2,8,8]
        args.nonblind_denoising = False

    elif task in ['004_RVRT_videodeblurring_DVD_16frames']:
        model = net(upscale=1, clip_size=2, img_size=[2, 64, 64], window_size=[2, 8, 8], num_blocks=[1, 2, 1],
                    depths=[2, 2, 2], embed_dims=[192, 192, 192], num_heads=[6, 6, 6],
                    inputconv_groups=[1, 3, 3, 3, 3, 3], deformable_groups=12, attention_heads=12,
                    attention_window=[3, 3], cpu_cache_length=100)
        datasets = ['DVD10']
        args.scale = 1
        args.window_size = [2,8,8]
        args.nonblind_denoising = False

    elif task in ['005_RVRT_videodeblurring_GoPro_16frames']:
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


    elif task in ["denoising","denoise_davis",'008_VRT_videodenoising_DAVIS',"rgb_denoise"]:
        model = net(upscale=1, clip_size=2, img_size=[2, 64, 64],
                    window_size=[2, 8, 8], num_blocks=[1, 2, 1],
                    depths=[2, 2, 2], embed_dims=[192, 192, 192], num_heads=[6, 6, 6],
                    inputconv_groups=[1, 3, 4, 6, 8, 4], deformable_groups=12,
                    attention_heads=12, attention_window=[3, 3],
                    nonblind_denoising=True, cpu_cache_length=100)
        datasets = ['Set8', 'DAVIS-test']
        args.scale = 1
        args.window_size = [2,8,8]
        args.nonblind_denoising = True
    return model,datasets,args

# def init_from_task(task,**kwargs):
#     ''' prepare model and dataset according to args.task. '''

#     # define model
#     args = edict()
#     if task == '001_VRT_videosr_bi_REDS_6frames':
#         model = net(upscale=4, img_size=[6,64,64], window_size=[6,8,8], depths=[8,8,8,8,8,8,8, 4,4,4,4, 4,4],
#                     indep_reconsts=[11,12], embed_dims=[120,120,120,120,120,120,120, 180,180,180,180, 180,180],
#                     num_heads=[6,6,6,6,6,6,6, 6,6,6,6, 6,6], pa_frames=2, deformable_groups=12)
#         datasets = ['REDS4']
#         args.scale = 4
#         args.window_size = [6,8,8]
#         args.nonblind_denoising = False

#     elif task == '002_VRT_videosr_bi_REDS_16frames':
#         model = net(upscale=4, img_size=[16,64,64], window_size=[8,8,8], depths=[8,8,8,8,8,8,8, 4,4,4,4, 4,4],
#                     indep_reconsts=[11,12], embed_dims=[120,120,120,120,120,120,120, 180,180,180,180, 180,180],
#                     num_heads=[6,6,6,6,6,6,6, 6,6,6,6, 6,6], pa_frames=6, deformable_groups=24)
#         datasets = ['REDS4']
#         args.scale = 4
#         args.window_size = [8,8,8]
#         args.nonblind_denoising = False

#     elif task in ['003_VRT_videosr_bi_Vimeo_7frames', '004_VRT_videosr_bd_Vimeo_7frames']:
#         model = net(upscale=4, img_size=[8,64,64], window_size=[8,8,8], depths=[8,8,8,8,8,8,8, 4,4,4,4, 4,4],
#                     indep_reconsts=[11,12], embed_dims=[120,120,120,120,120,120,120, 180,180,180,180, 180,180],
#                     num_heads=[6,6,6,6,6,6,6, 6,6,6,6, 6,6], pa_frames=4, deformable_groups=16)
#         datasets = ['Vid4'] # 'Vimeo'. Vimeo dataset is too large. Please refer to #training to download it.
#         args.scale = 4
#         args.window_size = [8,8,8]
#         args.nonblind_denoising = False

#     elif task in ["deblur_dvd",'005_VRT_videodeblurring_DVD']:
#         model = net(upscale=1, img_size=[6,192,192], window_size=[6,8,8], depths=[8,8,8,8,8,8,8, 4,4, 4,4],
#                     indep_reconsts=[9,10], embed_dims=[96,96,96,96,96,96,96, 120,120, 120,120],
#                     num_heads=[6,6,6,6,6,6,6, 6,6, 6,6], pa_frames=2, deformable_groups=16)
#         datasets = ['DVD10']
#         args.scale = 1
#         args.window_size = [6,8,8]
#         args.nonblind_denoising = False

#     elif task in ["deblur_gopro",'006_VRT_videodeblurring_GoPro']:
#         model = net(upscale=1, img_size=[6,192,192], window_size=[6,8,8], depths=[8,8,8,8,8,8,8, 4,4, 4,4],
#                     indep_reconsts=[9,10], embed_dims=[96,96,96,96,96,96,96, 120,120, 120,120],
#                     num_heads=[6,6,6,6,6,6,6, 6,6, 6,6], pa_frames=2, deformable_groups=16)
#         datasets = ['GoPro11-part1', 'GoPro11-part2']
#         args.scale = 1
#         args.window_size = [6,8,8]
#         args.nonblind_denoising = False

#     elif task in ["deblur_reds",'007_VRT_videodeblurring_REDS']:
#         model = net(upscale=1, img_size=[6,192,192], window_size=[6,8,8], depths=[8,8,8,8,8,8,8, 4,4, 4,4],
#                     indep_reconsts=[9,10], embed_dims=[96,96,96,96,96,96,96, 120,120, 120,120],
#                     num_heads=[6,6,6,6,6,6,6, 6,6, 6,6], pa_frames=2, deformable_groups=16)
#         datasets = ['REDS4']
#         args.scale = 1
#         args.window_size = [6,8,8]
#         args.nonblind_denoising = False

#     elif task in ["denoising","denoise_davis",'008_VRT_videodenoising_DAVIS',"rgb_denoise"]:
#         model = net(upscale=1, img_size=[6,192,192], window_size=[6,8,8],
#                     depths=[8,8,8,8,8,8,8, 4,4, 4,4],
#                     indep_reconsts=[9,10],
#                     embed_dims=[96,96,96,96,96,96,96, 120,120, 120,120],
#                     num_heads=[6,6,6,6,6,6,6, 6,6, 6,6], pa_frames=2,
#                     deformable_groups=16,
#                     nonblind_denoising=True)
#         datasets = ['Set8', 'DAVIS-test']
#         args.scale = 1
#         args.window_size = [6,8,8]
#         args.nonblind_denoising = True
#     elif task == '009_VRT_videofi_Vimeo_4frames':
#         model = net(upscale=1, out_chans=3, img_size=[4,192,192], window_size=[4,8,8], depths=[8,8,8,8,8,8,8, 4,4, 4,4],
#                     indep_reconsts=[], embed_dims=[96,96,96,96,96,96,96, 120,120, 120,120],
#                     num_heads=[6,6,6,6,6,6,6, 6,6, 6,6], pa_frames=0)
#         datasets = ['UCF101', 'DAVIS-train']  # 'Vimeo'. Vimeo dataset is too large. Please refer to #training to download it.
#         args.scale = 1
#         args.window_size = [4,8,8]
#         args.nonblind_denoising = False
#     else:
#         raise ValueError(f"Uknown task [{task}]")
#     return model,datasets,args
