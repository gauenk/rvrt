
# -- basic --
from pathlib import Path
from easydict import EasyDict as edict

# -- network --
from .network_rvrt import RVRT as net
from dev_basics import arch_io

# -- configs --
from dev_basics.configs import ExtractConfig,dcat
econfig = ExtractConfig(__file__) # init extraction
extract_config = econfig.extract_config # rename extraction

@econfig.set_init
def load_model(cfg):

    # -- unpack local vars --
    econfig.init(cfg)
    device = econfig.optional(cfg,"device","cuda")
    local_pairs = {"io":io_pairs(),
                   "imodel":{"task":"denoising","spynet_path":None,
                             "offset_type":"default",
                             "fixed_offset_max":2.5,
                             "attention_window":[3,3],
                             "offset_ws":3,"offset_stride1":0.5,
                             "offset_ps":1,"offset_dtype":"l2"}}
    cfgs = econfig.extract_dict_of_pairs(cfg,local_pairs,restrict=True)
    cfg = dcat(cfg,econfig.flatten(cfgs)) # update cfg
    if econfig.is_init: return

    # -- load net --
    model,datasets,args = get_model(cfgs.imodel)

    # -- load model --
    load_pretrained(model,cfgs.io)

    return model

def io_pairs():
    base = Path("weights/checkpoints/")
    pretrained_path = base / "model/model_best.pt"
    pairs = {"pretrained_load":False,
             "pretrained_path":str(pretrained_path),
             "pretrained_type":"lit",
             "pretrained_root":"."}
    return pairs


def load_pretrained(model,cfg):
    if cfg.pretrained_load:
        print("Loading model: ",cfg.pretrained_path)
        arch_io.load_checkpoint(model,cfg.pretrained_path,
                                cfg.pretrained_root,cfg.pretrained_type)

def get_model(cfg):

    ''' prepare model and dataset according to args.task. '''
    task = cfg.task
    spynet_path = cfg.spynet_path
    args = edict()
    args.task = task

    # define model
    if task == '001_RVRT_videosr_bi_REDS_30frames':
        model = net(upscale=4, clip_size=2, img_size=[2, 64, 64], window_size=[2, 8, 8], num_blocks=[1, 2, 1],
                    depths=[2, 2, 2], embed_dims=[144, 144, 144], num_heads=[6, 6, 6],
                    inputconv_groups=[1, 1, 1, 1, 1, 1], deformable_groups=12, attention_heads=12,
                    attention_window=[3, 3], cpu_cache_length=100,
                    spynet_path=spynet_path,offset_type=cfg.offset_type,
                    fixed_offset_max=cfg.fixed_offset_max,
                    offset_ws=cfg.offset_ws,offset_ps=cfg.offset_ps,
                    offset_stride1=cfg.offset_stride1,offset_dtype=cfg.offset_dtype)
        datasets = ['REDS4']
        args.scale = 4
        args.window_size = [2,8,8]
        args.nonblind_denoising = False

    elif task in ["sr",'002_RVRT_videosr_bi_Vimeo_14frames', '003_RVRT_videosr_bd_Vimeo_14frames']:
        model = net(upscale=4, clip_size=2, img_size=[2, 64, 64], window_size=[2, 8, 8], num_blocks=[1, 2, 1],
                    depths=[2, 2, 2], embed_dims=[144, 144, 144], num_heads=[6, 6, 6],
                    inputconv_groups=[1, 1, 1, 1, 1, 1], deformable_groups=12, attention_heads=12,
                    attention_window=[3, 3], cpu_cache_length=100,
                    spynet_path=spynet_path,offset_type=cfg.offset_type,
                    fixed_offset_max=cfg.fixed_offset_max,
                    offset_ws=cfg.offset_ws,offset_ps=cfg.offset_ps,
                    offset_stride1=cfg.offset_stride1,offset_dtype=cfg.offset_dtype)
        datasets = ['Vid4'] # 'Vimeo'. Vimeo dataset is too large. Please refer to #training to download it.
        args.scale = 4
        args.window_size = [2,8,8]
        args.nonblind_denoising = False

    elif task in ['004_RVRT_videodeblurring_DVD_16frames']:
        model = net(upscale=1, clip_size=2, img_size=[2, 64, 64], window_size=[2, 8, 8], num_blocks=[1, 2, 1],
                    depths=[2, 2, 2], embed_dims=[192, 192, 192], num_heads=[6, 6, 6],
                    inputconv_groups=[1, 3, 3, 3, 3, 3], deformable_groups=12, attention_heads=12,
                    attention_window=[3, 3], cpu_cache_length=100,
                    spynet_path=spynet_path,offset_type=cfg.offset_type,
                    fixed_offset_max=cfg.fixed_offset_max,
                    offset_ws=cfg.offset_ws,offset_ps=cfg.offset_ps,
                    offset_stride1=cfg.offset_stride1,offset_dtype=cfg.offset_dtype)
        datasets = ['DVD10']
        args.scale = 1
        args.window_size = [2,8,8]
        args.nonblind_denoising = False

    elif task in ["gopro+deblur",'005_RVRT_videodeblurring_GoPro_16frames']:
        model = net(upscale=1, clip_size=2, img_size=[2, 64, 64],
                    window_size=[2, 8, 8], num_blocks=[1, 2, 1],
                    depths=[2, 2, 2], embed_dims=[192, 192, 192], num_heads=[6, 6, 6],
                    inputconv_groups=[1, 3, 3, 3, 3, 3], deformable_groups=12,
                    attention_heads=12,
                    attention_window=[3, 3], cpu_cache_length=100,
                    spynet_path=spynet_path,offset_type=cfg.offset_type,
                    fixed_offset_max=cfg.fixed_offset_max,
                    offset_ws=cfg.offset_ws,offset_ps=cfg.offset_ps,
                    offset_stride1=cfg.offset_stride1,offset_dtype=cfg.offset_dtype)
        datasets = ['GoPro11-part1', 'GoPro11-part2']
        args.scale = 1
        args.window_size = [2,8,8]
        args.nonblind_denoising = False

    elif task in ["denoising","denoise_davis",'008_VRT_videodenoising_DAVIS',"rgb_denoise"]:
        model = net(upscale=1, clip_size=2, img_size=[2, 64, 64],
                    window_size=[2, 8, 8], num_blocks=[1, 2, 1],
                    depths=[2, 2, 2], embed_dims=[192, 192, 192], num_heads=[6, 6, 6],
                    inputconv_groups=[1, 3, 4, 6, 8, 4], deformable_groups=12,
                    attention_heads=12, attention_window=cfg.attention_window,
                    nonblind_denoising=True, cpu_cache_length=100,
                    spynet_path=spynet_path,offset_type=cfg.offset_type,
                    fixed_offset_max=cfg.fixed_offset_max,
                    offset_ws=cfg.offset_ws,offset_ps=cfg.offset_ps,
                    offset_stride1=cfg.offset_stride1,offset_dtype=cfg.offset_dtype)
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
