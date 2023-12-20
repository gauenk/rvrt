"""

  Inspect statistics of the offsets computed from the DAVIS dataset
  using the trained RVRT model

"""

# -- basic --
import numpy as np
import torch as th
from einops import rearrange
from torchvision.utils import save_image,make_grid
from typing import Any, Callable
from easydict import EasyDict as edict
from dev_basics.utils.misc import ensure_chnls
from dev_basics.utils.metrics import compute_psnrs,compute_ssims,compute_strred

# -- plotting --
import seaborn as sns
import matplotlib.pyplot as plt

# -- better res --
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

# -- forward processing --
from dev_basics import net_chunks
from dev_basics.utils import vid_io
from dev_basics.utils.misc import set_seed

# -- data --
import data_hub

# -- extra --
from stnls.dev.misc.viz_nls_map import get_search_grid,search_deltas,bound

# -- extract config --
from dev_basics.configs import ExtractConfig,dcat
econfig = ExtractConfig(__file__)

def viz_offsets(dmap,offs,ws,stride1,imgH,imgW):
    O = len(offs)
    dmap /= dmap.max()
    H,W = dmap.shape[-2:]
    # r = stride1 / net_stride1
    # print("offset viz.")
    # print(offs)
    for o,_off in enumerate(offs):
        _off0 = _off[0].item()
        _off1 = _off[1].item()
        # print(_off0,_off1)
        # _off0 = bound(_off0,imgH)
        # _off1 = bound(_off1,imgW)

        wo = 1.#np.exp(-(1/10.)*o/O)
        off = [_off0/stride1 + (ws-1)//2,_off1/stride1 + (ws-1)//2]
        # print(off,stride1)
        off_i = [off[0],off[1]]
        # print(off_i,_off)

        # -- nearest index --
        off_i[0] = round(off[0])
        off_i[1] = round(off[1])
        off_i[0] = bound(off_i[0],H)
        off_i[1] = bound(off_i[1],W)
        if off_i[0] > (H-1) or off_i[0] < 0:
            # print("skip.")
            continue
        if off_i[1] > (W-1) or off_i[1] < 0:
            # print("skip.")
            continue
        dmap[:,off_i[0],off_i[1]] = 0
        dmap[2,off_i[0],off_i[1]] = 1

    return dmap

def viz_offsets_bilin2d(dmap,offs,ws,stride1):
    O = len(offs)
    dmap /= dmap.max()
    H,W = dmap.shape[-2:]
    # r = stride1 / net_stride1
    # print("offset viz.")
    # print(offs)
    for o,_off in enumerate(offs):
        wo = 1.#np.exp(-(1/10.)*o/O)
        off = [_off[0].item()/stride1 + (ws-1)//2,_off[1].item()/stride1 + (ws-1)//2]
        # print(off,stride1)
        off_i = [off[0],off[1]]
        # print(off_i,_off)
        Z = 0
        for ix in range(2):
            for jx in range(2):
                off_i[0] = round(off[0] + ix)
                off_i[1] = round(off[1] + jx)
                wi = max(0,1-abs(off_i[0] - off[0]))
                wj = max(0,1-abs(off_i[1] - off[1]))
                off_i[0] = bound(off_i[0],H)
                off_i[1] = bound(off_i[1],W)
                if off_i[0] > (H-1) or off_i[0] < 0: continue
                if off_i[1] > (W-1) or off_i[1] < 0: continue
                w = wi*wj
                if w > 1e-1:
                    dmap[:,off_i[0],off_i[1]] = 0
                    Z += w

        for ix in range(2):
            for jx in range(2):
                off_i[0] = round(off[0] + ix)
                off_i[1] = round(off[1] + jx)
                wi = max(0,1-abs(off_i[0] - off[0]))
                wj = max(0,1-abs(off_i[1] - off[1]))
                off_i[0] = bound(off_i[0],H)
                off_i[1] = bound(off_i[1],W)
                if off_i[0] > (H-1) or off_i[0] < 0: continue
                if off_i[1] > (W-1) or off_i[1] < 0: continue
                w = wi*wj
                if w > 1e-1:
                    dmap[2,off_i[0],off_i[1]] += w*wo/Z
    # dmap /= dmap.max()
    return dmap

class OffsetInfoHook():

    def __init__(self,net):
        self.net = net
        self.net_stride1 = net.deform_align.forward_1.stride
        self.net_ws = net.deform_align.forward_1.stride
        self.dist_type = net.deform_align.forward_1.offset_dtype
        self.ps = net.deform_align.forward_1.offset_ps

        # -- register buffers as dicts --
        buf_list = ["o1_offset_ftrs","o2_offset_ftrs",
                    "q_ftrs","k_ftrs",
                    "flow_ftrs"]
        for buf in buf_list:
            setattr(self,buf,{})

        # -- register hooks with buffer names --
        for name,layer in self.net.named_modules():
            for buf in buf_list:
                layer_suffix = buf.replace("ftrs","shell")
                if name.endswith(layer_suffix):
                    layer.register_forward_hook(self.save_outputs_hook(buf,name))

    def save_outputs_hook(self, buffer_name: str, layer_id: str) -> Callable:
        buff = getattr(self,buffer_name)
        def fn(_, __, output):
            if layer_id in buff:
                buff[layer_id].append(output)
            else:
                buff[layer_id] = [output]
        return fn

    def summary(self):
        for name in self.o1_offset_ftrs:
            ftrs = self.o1_offset_ftrs[name]
            fmin = min([f.min().item() for f in ftrs])
            fmax = max([f.max().item() for f in ftrs])
            fmean = np.mean([f.mean().item() for f in ftrs])
            print(name,fmin,fmean,fmax)

    def hist(self,fn="hist"):

        # -- collect history --
        agg = []
        for name in self.o1_offset_ftrs:
            ftrs_l = self.o1_offset_ftrs[name]
            for ftrs in ftrs_l:
                _ftrs = ftrs[0].reshape(-1,2,64*64).transpose(1,0).reshape(2,-1)
                agg.append(_ftrs)
        agg = th.cat(agg,-1)
        # print(agg.shape)

        def subsample(agg,N):
            inds = th.randperm(agg.shape[1])[:N]
            agg = agg[:,inds]
            return agg

        # -- plot sample --
        N = 10**4
        num_hists = 3
        for hist in range(num_hists):
            agg_sub = subsample(agg,N).cpu().numpy().T
            fig,ax = plt.subplots()
            sns.kdeplot(agg_sub,ax=ax)
            ax.set_xlim([-10,10])
            # ax.hist(agg_sub)
            fn_sub = "%s_%02d.png" % (fn,hist)
            plt.savefig(fn_sub)
            plt.close("all")
            plt.clf()

    def ishow(self,H,W,fn="ishow.png"):
        i,j = 32,32
        F = len(self.o1_offset_ftrs)
        vid = th.zeros((F,1,H,W),device="cuda:0")
        ps = 3
        for f,name in enumerate(self.o1_offset_ftrs):
            ftrs_l = self.o1_offset_ftrs[name]
            for ftrs in ftrs_l:
                _ftrs = ftrs[0,:,:,i,j].round().int()
                for fi in _ftrs:
                    # print(fi[0],fi[1])
                    vid[...,i+fi[0]-ps//2:i+fi[0]+ps//2,
                        j+fi[1]-ps//2:j+fi[1]+ps//2] += 1.
        vid = vid/vid.max()
        save_image(nicer_image(vid),fn)

    def show_dmap(self,fn="dmap.png"):

        # -- imports --

        # -- config --
        ps = self.ps
        ws = 41
        stride1 = 0.5

        # -- get q,k --
        i = 1
        fmt = "deform_align.forward_1.%s_shell"
        name = fmt % "q"
        qvid = self.q_ftrs[name][i]
        name = fmt % "k"
        kvid = self.k_ftrs[name][i]
        name = fmt % "flow"
        flows = self.flow_ftrs[name][i]
        name = fmt % "o1_offset"
        offset1_ftrs = self.o1_offset_ftrs[name][i]
        name = fmt % "o2_offset"
        offset2_ftrs = self.o2_offset_ftrs[name][i]

        # -- prepare video for search --
        # print(len(flows),len(flows[0]),flows[0].shape)
        # print(len(qvid),qvid[0].shape)

        # -- search order to match rvrt --
        qorder = [0,1,1,0]
        korder = [0,1,0,1]
        forder = [[0,0],[0,1],[1,0],[1,1]]
        n = 0

        # -- prepare data --
        zvid = th.inf*th.ones_like(qvid[:,[0]])
        zflow = th.zeros_like(flows[0][:,[0]])
        bflow = th.zeros_like(flows[0]).cpu()
        qvid_n = qvid[:,[qorder[n]]]
        kvid_n = kvid[:,[korder[n]]]
        fflow_n = flows[forder[n][0]][:,[forder[n][1]]]
        qvid_n = th.cat([qvid_n,zvid],1).cpu()
        kvid_n = th.cat([-zvid,kvid_n],1).cpu()
        fflow_n = th.cat([fflow_n,zflow],1).cpu()
        # print(qvid_n.shape,kvid_n.shape,fflow_n.shape)

        # -- index first batch --
        qvid_n = qvid_n[0]
        kvid_n = kvid_n[0]
        fflow_n = fflow_n[0]

        # -- add heads --
        nheads = 12
        qvid_n = rearrange(qvid_n,'b (hd c) h w -> hd b c h w',hd=nheads)
        kvid_n = rearrange(kvid_n,'b (hd c) h w -> hd b c h w',hd=nheads)

        # -- compute map --
        grid = get_search_grid(ws)
        # loc0 = [0,0,0]
        loc0 = [0,25,25]
        dmaps = []
        for i in range(6):#nheads):
            dmap_i = search_deltas(qvid_n[i],kvid_n[i],fflow_n,bflow,
                                   loc0,grid,stride1,ws,ps,dist_type=self.dist_type)
            # save_image(dmap_i,"dmap_%d.png" % i)
            dmaps.append(dmap_i)
        dmaps = th.stack(dmaps)

        # -- from offsets to indices --
        H,W = offset1_ftrs.shape[-2:]
        off1,off2 = offset1_ftrs[0],offset2_ftrs[0]
        off1 = rearrange(off1,'t (HD k two) h w -> t HD k two h w',HD=nheads,two=2)
        off2 = off2.reshape(2,nheads,9,2,H,W)
        inds0,inds1 = off1[0],off1[1]
        inds2,inds3 = off2[0],off2[1]

        # print("inds0.shape: ",inds0.shape)
        omaps = []
        for i in range(6):#nheads):
            omap_i = viz_offsets(dmaps[i].clone(),
                                 inds0[i,:,:,loc0[1]-1,loc0[2]-1],ws,stride1,H,W)
            omaps.append(omap_i)
            # save_image(omap_i,"omap_%d.png" % i)
        omaps = th.stack(omaps)

        # -- dmaps,omaps --
        # print(dmaps.shape,omaps.shape)
        # maps = th.stack([dmaps,omaps],0)
        maps = omaps[None,:]
        # maps = maps[:,[1,3]]
        nrow = maps.shape[0]
        maps = maps.transpose(0,1).flatten(0,1)
        grid = make_grid(maps,nrow=nrow,pad_value=1.)
        grid = grid[...,2:-2,2:-2] # remove exterior padding
        save_image(nicer_image(grid),"grid.png")

        # print(inds0.shape)
        # print(inds0[...,32,32])

        # -- create grid --
        # grid_y, grid_x = torch.meshgrid(torch.arange(0, H, dtype=dtype, device=device),
        #                                 torch.arange(0, W, dtype=dtype, device=device))
        # grid = torch.stack((grid_y, grid_x), 2).float()  # W(x), H(y), 2
        # grid = rearrange(grid,'H W two -> two H W')
        # print(inds0.shape)
        # inds0 = inds0 + grid[None,]
        # inds0 = inds0 + fflow_n[[0]].flip(-3)



def run_exp(cfg):

    # -- init config --
    econfig.init(cfg)
    net_module = econfig.required_module(cfg,"python_module")
    net_extract_config = net_module.extract_config
    cfgs = econfig.extract_set({"net":net_extract_config(cfg)})
    cfg = dcat(cfg,econfig.flatten(cfgs)) # update cfg
    if econfig.is_init: return
    device = cfg.device
    imax = 255.
    set_seed(cfg.seed)

    # -- load values --
    net = net_module.load_model(cfgs.net).to(cfg.device)
    data,loaders = data_hub.sets.load(cfg)
    indices = data_hub.filter_subseq(data[cfg.dset],cfg.vid_name,
                                     cfg.frame_start,cfg.frame_end)

    # -- iterate over sub videos --
    for index in indices:

        # -- clean memory --
        th.cuda.empty_cache()

        # -- unpack --
        sample = data[cfg.dset][index]
        region = sample['region']
        noisy,clean = sample['noisy'][None,],sample['clean'][None,]
        noisy,clean = noisy.to(cfg.device),clean.to(cfg.device)
        sample['sigma'] = sample['sigma'][None,].to(cfg.device)
        noisy = ensure_chnls(cfg.dd_in,noisy,sample)
        vid_frames = sample['fnums'].numpy()
        # print("[%d] noisy.shape: " % index,noisy.shape)

        # vid_io.save_video(noisy,"output/testing/","noisy")
        # vid_io.save_video(clean,"output/testing/","clean")

        # -- downsample if noisy sr --
        if (cfg.task == "sr"):
            scale = 0.25
            H,W = noisy.shape[-2:]
            cH,cW = int(scale*H),int(scale*W)
            B = noisy.shape[0]
            noisy = rearrange(noisy,'b t ... -> (b t) ...')
            noisy = TF.resize(noisy,(cH,cW),InterpolationMode.BILINEAR)
            noisy = rearrange(noisy,'(b t) ... -> b t ...',b=B)
            # noisy = tvF.interpolate(noisy,scale=0.5,mode="nearest")

        # -- add hooks --
        hook = OffsetInfoHook(net)

        # -- forward --
        chunk_cfg = net_chunks.extract_chunks_config(cfg)
        fwd_fxn = net_chunks.chunk(chunk_cfg,net.forward)
        with th.no_grad():
            deno = fwd_fxn(noisy/imax,None)*imax
        # vid_io.save_video(deno,"output/testing/","deno_srch")
        # vid_io.save_video(deno,"output/testing/","deno_def")


        # -- ensure working
        psnrs = compute_psnrs(clean,deno,div=imax)
        ssims = compute_ssims(clean,deno,div=imax)
        strred = compute_strred(clean,deno,div=imax)
        print("psnrs: ",np.mean(psnrs))
        print("ssims: ",np.mean(ssims))
        print("strred: ",np.mean(strred))

        # -- compute offset info --
        # H,W = noisy.shape[-2:]
        H,W = 256,256
        print(hook.summary())
        # print(hook.ishow(H//4,W//4))
        # print(hook.hist())
        hook.show_dmap()

def nicer_image(vid):
    B = vid.shape[0]
    H,W = vid.shape[-2:]
    cH = 6*H
    cW = 6*W
    ndim = vid.ndim
    if ndim == 5:
        vid = rearrange(vid,'b t ... -> (b t) ...')
    vid = TF.resize(vid,(cH,cW),InterpolationMode.NEAREST)
    if ndim == 5:
        vid = rearrange(vid,'(b t) ... -> b t ...',b=B)
    return vid

def main():

    cfg = edict()
    cfg.seed = 123
    cfg.device = "cuda:0"
    # cfg.offset_type = "fixed"
    # cfg.offset_type = "default"
    # cfg.offset_type = "search"
    cfg.offset_type = "refine"
    # cfg.offset_type = "search"
    cfg.offset_dtype = "prod"
    # cfg.offset_dtype = "l2"
    cfg.fixed_offset_max = 2.5
    cfg.attention_window = [3,3]
    cfg.python_module = "rvrt"
    cfg.dname = "set8"
    cfg.nframes = 6
    cfg.frame_start = 0
    cfg.frame_end = 5
    # cfg.isize = None
    cfg.isize = "512_512"
    cfg.spatial_chunk_size = 512
    cfg.spatial_chunk_overlap = 0.25
    cfg.temporal_chunk_size = 6
    cfg.temporal_chunk_overlap = 0.25
    cfg.vid_name = "sunflower"
    cfg.dset = "te"
    cfg.sigma = 0.01
    cfg.pretrained_root = "."
    cfg.pretrained_type = "git"
    cfg.pretrained_load = True

    # cfg.sigma = 60
    # cfg.offset_wr = 1
    # cfg.offset_ws = 3
    # cfg.offset_stride1 = 0.5
    # cfg.pretrained_path = "weights/006_RVRT_videodenoising_DAVIS_16frames.pth"
    # cfg.task = "denoising"
    # cfg.dd_in = 4

    cfg.sigma = 0.01
    cfg.offset_wr = 1
    cfg.offset_ws = 9
    cfg.offset_stride1 = 1.
    cfg.pretrained_path = "weights/002_RVRT_videosr_bi_Vimeo_14frames.pth"
    cfg.task = "sr"
    cfg.dd_in = 3

    run_exp(cfg)

if __name__ == "__main__":
    main()
