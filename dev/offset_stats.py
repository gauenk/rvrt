"""

  Inspect statistics of the offsets computed from the DAVIS dataset
  using the trained RVRT model

"""

# -- basic --
import numpy as np
import torch as th
from torchvision.utils import save_image
from typing import Any, Callable
from easydict import EasyDict as edict
from dev_basics.utils.misc import ensure_chnls
from dev_basics.utils.metrics import compute_psnrs,compute_ssims,compute_strred

# -- plotting --
import seaborn as sns
import matplotlib.pyplot as plt

# -- forward processing --
from dev_basics import net_chunks

# -- data --
import data_hub

# -- extract config --
from dev_basics.configs import ExtractConfig,dcat
econfig = ExtractConfig(__file__)


class OffsetInfoHook():

    def __init__(self,net):
        self.net = net
        self.features = {}
        # -- register hooks --
        for name,layer in self.net.named_modules():
            # if not(name.endswith("conv_offset")): continue
            if not(name.endswith("offset_shell")): continue
            layer.register_forward_hook(self.save_outputs_hook(name))
        # exit()

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            if layer_id in self.features:
                self.features[layer_id].append(output)
            else:
                self.features[layer_id] = [output]
        return fn

    def summary(self):
        for name in self.features:
            ftrs = self.features[name]
            fmin = min([f.min().item() for f in ftrs])
            fmax = max([f.max().item() for f in ftrs])
            fmean = np.mean([f.mean().item() for f in ftrs])
            print(name,fmin,fmean,fmax)

    def hist(self,fn="hist"):

        # -- collect history --
        agg = []
        for name in self.features:
            ftrs_l = self.features[name]
            for ftrs in ftrs_l:
                _ftrs = ftrs[0].reshape(-1,2,64*64).transpose(1,0).reshape(2,-1)
                agg.append(_ftrs)
        agg = th.cat(agg,-1)
        print(agg.shape)

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
        F = len(self.features)
        vid = th.zeros((F,1,H,W),device="cuda:0")
        ps = 3
        for f,name in enumerate(self.features):
            ftrs_l = self.features[name]
            for ftrs in ftrs_l:
                _ftrs = ftrs[0,:,:,i,j].round().int()
                for fi in _ftrs:
                    # print(fi[0],fi[1])
                    vid[...,i+fi[0]-ps//2:i+fi[0]+ps//2,
                        j+fi[1]-ps//2:j+fi[1]+ps//2] += 1.
        vid = vid/vid.max()
        save_image(vid,fn)

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
        print("[%d] noisy.shape: " % index,noisy.shape)

        # -- add hooks --
        hook = OffsetInfoHook(net)

        # -- forward --
        chunk_cfg = net_chunks.extract_chunks_config(cfg)
        fwd_fxn = net_chunks.chunk(chunk_cfg,net.forward)
        with th.no_grad():
            deno = fwd_fxn(noisy/imax,None)*imax

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

def main():

    cfg = edict()
    cfg.device = "cuda:0"
    cfg.offset_type = "fixed"
    cfg.fixed_offset_max = 0.01
    cfg.attention_window = [3,3]
    cfg.python_module = "rvrt"
    cfg.dname = "set8"
    cfg.nframes = 6
    cfg.frame_start = 0
    cfg.frame_end = 5
    cfg.isize = None
    cfg.spatial_chunk_size = 256
    cfg.spatial_chunk_overlap = 0.25
    cfg.temporal_chunk_size = 6
    cfg.temporal_chunk_overlap = 0.25
    cfg.vid_name = "sunflower"
    cfg.dd_in = 4
    cfg.dset = "te"
    cfg.sigma = 50
    cfg.pretrained_root = "."
    cfg.pretrained_type = "git"
    cfg.pretrained_path = "weights/006_RVRT_videodenoising_DAVIS_16frames.pth"
    cfg.pretrained_load = True

    run_exp(cfg)

if __name__ == "__main__":
    main()
