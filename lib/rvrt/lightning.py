

# -- misc --
import os,math,tqdm,sys
import pprint,copy
pp = pprint.PrettyPrinter(indent=4)

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- data mngmnt --
from pathlib import Path
from easydict import EasyDict as edict

# -- vision --
import torchvision.transforms as tvt

# -- data --
import data_hub

# -- optical flow --
from dev_basics import flow

# -- caching results --
import cache_io

# # -- network --
# import nlnet

# -- configs --
from dev_basics.configs import ExtractConfig
econfig = ExtractConfig(__file__)
extract_config = econfig.extract_config

# -- misc --
from dev_basics.utils.misc import rslice,write_pickle,read_pickle
from dev_basics.utils.metrics import compute_psnrs,compute_ssims
from dev_basics.utils.timer import ExpTimer
import dev_basics.utils.gpu_mem as gpu_mem

# -- noise sims --
import importlib
# try:
#     import stardeno
# except:
#     pass

# # -- wandb --
# WANDB_AVAIL = False
# try:
#     import wandb
#     WANDB_AVAIL = True
# except:
#     pass

# -- generic logging --
import logging
logging.basicConfig()

# -- lightning module --
import torch
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_only

# import torch
# torch.autograd.set_detect_anomaly(True)

@econfig.set_init
def init_cfg(cfg):
    econfig.init(cfg)
    cfgs = econfig.extract_dict_of_pairs(cfg,{"lit":lit_pairs(),
                                              "sim":sim_pairs()},
                                         restrict=True)
    return cfgs

def lit_pairs():
    pairs = {"batch_size":1,"flow":True,"flow_method":"cv2",
             "isize":None,"bw":False,"lr_init":1e-3,
             "lr_final":1e-8,"weight_decay":1e-8,
             "nepochs":0,"nsteps":0,"task":"denoising",
             "uuid":"","scheduler_name":"default",
             "step_lr_size":5,"step_lr_gamma":0.1,
             "flow_epoch":None,"flow_from_end":None,
             "use_wandb":False,"ntype":"g","rate":-1,"sigma":-1,
             "sigma_min":-1,"sigma_max":-1,
             "optim_name":"adamw",
             "sgd_momentum":0.1,"sgd_dampening":0.1,
             "coswr_T0":-1,"coswr_Tmult":1,"coswr_eta_min":1e-9,
             "step_lr_multisteps":"30-50",
             "spynet_global_step":-1,"limit_train_batches":-1,"dd_in":4,
             "fill_loss":False,"fill_loss_weight":1.,"fill_loss_n":10,
             "fill_loss_scale_min":.01,"fill_loss_scale_max":0.05}
    return pairs

def sim_pairs():
    pairs = {"sim_type":"g","sim_module":"stardeno",
             "sim_device":"cuda:0","load_fxn":"load_sim"}
    return pairs

def get_sim_model(self,cfg):
    if cfg.sim_type == "g":
        return None
    elif cfg.sim_type == "stardeno":
        module = importlib.load_module(cfg.sim_module)
        return module.load_noise_sim(cfg.sim_device,True).to(cfg.sim_device)
    else:
        raise ValueError(f"Unknown sim model [{sim_type}]")

class LitModel(pl.LightningModule):

    def __init__(self,lit_cfg,net,sim_model):
        super().__init__()
        lit_cfg = init_cfg(lit_cfg).lit
        for key,val in lit_cfg.items():
            setattr(self,key,val)
        self.set_flow_epoch()
        self.net = net
        self.sim_model = sim_model
        self.gen_loger = logging.getLogger('lightning')
        self.gen_loger.setLevel("NOTSET")
        self.automatic_optimization=True

    def forward(self,vid,flows=None):
        if flows is None:
            flows = flow.orun(vid,self.flow,ftype=self.flow_method)
        # print(vid.shape)
        deno = self.net(vid)#,flows=flows)
        return deno

    def sample_noisy(self,batch):
        if self.sim_model is None: return
        clean = batch['clean']
        noisy = self.sim_model.run_rgb(clean)
        batch['noisy'] = noisy

    def set_flow_epoch(self):
        if not(self.flow_epoch is None): return
        if self.flow_from_end is None: return
        if self.flow_from_end == 0: return
        self.flow_epoch = self.nepochs - self.flow_from_end

    def update_flow(self):
        if self.flow_epoch is None: return
        if self.flow_epoch <= 0: return
        if self.current_epoch >= self.flow_epoch:
            self.flow = True

    def get_params(self):

        #
        # -- basic parameters --
        #

        params = [{"params":self.parameters()}]
        if not(self.uses_spynet()):
            return params
        #
        # -- include spynet with separate learning rate --
        #

        # -- all parameters except spynet --
        named_params = self.net.named_parameters()
        base_params = list(filter(lambda kv: not("spynet" in kv[0]), named_params))
        base_params = [kv[1] for kv in base_params]
        # print(base_params[0])

        # -- spynet params --
        spynet_params = self.net.spynet.parameters()
        # print(list(spynet_params)[0])
        params = [{"params":base_params},
                  {'params': spynet_params, 'lr': self.lr_init*0.25}]
        return params

    def configure_optimizers(self):
        params = self.get_params()
        if self.optim_name == "adam":
            optim = th.optim.Adam(params,lr=self.lr_init,
                                  weight_decay=self.weight_decay)
        elif self.optim_name == "adamw":
            optim = th.optim.AdamW(params,lr=self.lr_init,
                                   weight_decay=self.weight_decay)
        elif self.optim_name == "sgd":
            optim = th.optim.SGD(params,lr=self.lr_init,
                                 weight_decay=self.weight_decay,
                                 momentum=self.sgd_momentum,
                                 dampening=self.sgd_dampening)
        else:
            raise ValueError(f"Unknown optim [{self.optim_name}]")
        sched = self.configure_scheduler(optim)
        print(sched)
        return [optim], [sched]

    def configure_scheduler(self,optim):

        if self.scheduler_name in ["default","exp_decay"]:
            gamma = math.exp(math.log(self.lr_final/self.lr_init)/self.nepochs)
            ExponentialLR = th.optim.lr_scheduler.ExponentialLR
            scheduler = ExponentialLR(optim,gamma=gamma) # (.995)^50 ~= .78
            scheduler = {"scheduler": scheduler, "interval": "epoch", "frequency": 1}
        elif self.scheduler_name in ["step","steplr"]:
            args = (self.step_lr_size,self.step_lr_gamma)
            # print("[Scheduler]: StepLR(%d,%2.2f)" % args)
            StepLR = th.optim.lr_scheduler.StepLR
            scheduler = StepLR(optim,step_size=self.step_lr_size,
                               gamma=self.step_lr_gamma)
        elif self.scheduler_name in ["cosa"]:
            CosAnnLR = th.optim.lr_scheduler.CosineAnnealingLR
            scheduler = CosAnnLR(optim,self.nepochs)
            scheduler = {"scheduler": scheduler, "interval": "epoch", "frequency": 1}
        elif self.scheduler_name in ["cosa_step"]:
            nsteps = self.num_steps()
            print("[CosAnnLR] nsteps: ",nsteps)
            CosAnnLR = th.optim.lr_scheduler.CosineAnnealingLR
            scheduler = CosAnnLR(optim,T_max=nsteps)
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        elif self.scheduler_name in ["multi_step"]:
            milestones = [int(x) for x in self.step_lr_multisteps.split("-")]
            MultiStepLR = th.optim.lr_scheduler.MultiStepLR
            scheduler = MultiStepLR(optim,milestones=milestones,
                                    gamma=self.step_lr_gamma)
            scheduler = {"scheduler": scheduler, "interval": "epoch", "frequency": 1}
        elif self.scheduler_name in ["coswr","cosw"]:
            lr_sched =th.optim.lr_scheduler
            CosineAnnealingWarmRestarts = lr_sched.CosineAnnealingWarmRestarts
            # print(self.coswr_T0,self.coswr_Tmult,self.coswr_eta_min)
            scheduler = CosineAnnealingWarmRestarts(optim,self.coswr_T0,
                                                    T_mult=self.coswr_Tmult,
                                                    eta_min=self.coswr_eta_min)
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        elif self.scheduler_name in ["none"]:
            StepLR = th.optim.lr_scheduler.StepLR
            scheduler = StepLR(optim,step_size=10**3,gamma=1.)
        else:
            raise ValueError(f"Uknown scheduler [{self.scheduler_name}]")
        return scheduler

    def training_step(self, batch, batch_idx):

        # -- set spynet to training --
        # print("global_step: ",self.global_step,self.spynet_global_step)
        if (self.global_step == 0) or (self.global_step < self.spynet_global_step):
            if self.uses_spynet(): self.net.spynet.eval()
        elif self.global_step >= self.spynet_global_step:
            print("Fine-tuning Spynet training.")
            if self.uses_spynet(): self.net.spynet.train()
        # if self.global_step == self.spynet_global_step:
        #     if hasattr(self.net,"spynet"):
        #         self.net.spynet.train()

        # -- sample noise from simulator --
        self.sample_noisy(batch)

        # -- update flow --
        self.update_flow()
        # print(batch['noisy'][:,0,-1,0,0],batch['sigma'])

        # -- each sample in batch --
        loss = 0 # init @ zero
        denos,fills,cleans = [],[],[]
        ntotal = len(batch['noisy'])
        nbatch = ntotal
        nbatches = (ntotal-1)//nbatch+1
        for i in range(nbatches):
            start,stop = i*nbatch,min((i+1)*nbatch,ntotal)
            deno_i,fill_i,clean_i,loss_i = self.training_step_i(batch, start, stop)
            loss += loss_i
            denos.append(deno_i)
            fills.append(fill_i)
            cleans.append(clean_i)
        loss = loss / nbatches

        # -- view params --
        # loss.backward()
        # for name, param in self.net.named_parameters():
        #     if param.grad is None:
        #         print(name)

        # -- append --
        denos = th.cat(denos)
        cleans = th.cat(cleans)
        fills = th.cat(fills) if self.fill_loss else None

        # -- log --
        get_psnr = lambda x,y: np.mean(compute_psnrs(x,y,div=1.)).item()
        val_psnr = get_psnr(denos,cleans)
        fill_psnr = get_psnr(fills,cleans) if self.fill_loss else -1
        lr = self.optimizers()._optimizer.param_groups[0]['lr']
        lr1 = -1
        if len(self.optimizers()._optimizer.param_groups) > 1:
            lr1 = self.optimizers()._optimizer.param_groups[1]['lr']
        # val_ssim = np.mean(compute_ssims(denos,cleans,div=1.)).item() # too slow.
        self.log("train_loss", loss.item(), on_step=True,
                 on_epoch=False, batch_size=self.batch_size, sync_dist=False)
        self.log("train_psnr", val_psnr, on_step=True,
                 on_epoch=False, batch_size=self.batch_size, sync_dist=False)
        if self.fill_loss:
            self.log("fill_psnr", fill_psnr, on_step=True,
                     on_epoch=False, batch_size=self.batch_size, sync_dist=False)
        self.log("lr", lr, on_step=True,
                 on_epoch=False, batch_size=self.batch_size, sync_dist=False)
        if lr1 >= 0:
            self.log("lr1", lr1, on_step=True,
                     on_epoch=False, batch_size=self.batch_size, sync_dist=False)
        self.log("global_step", self.global_step, on_step=True,
                 on_epoch=False, batch_size=self.batch_size, sync_dist=False)
        # self.log("train_ssim", val_ssim, on_step=True,
        #          on_epoch=False, batch_size=self.batch_size)
        # self.gen_loger.info("train_psnr: %2.2f" % val_psnr)

        return loss

    def training_step_fill(self, clean, flows):

        # -- unpack batch
        tofill = clean.clone()
        T = tofill.shape[1]
        fill_loss_scale_min = self.fill_loss_scale_min
        fill_loss_scale_max = self.fill_loss_scale_max
        fill_loss_n = self.fill_loss_n

        # -- erase randomly --
        eraser = tvt.RandomErasing(p=1.,
                                   scale=(fill_loss_scale_min, fill_loss_scale_max),
                                   ratio=(0.3, 3.3), value=0, inplace=False)
        for t in range(T):
            for n in range(fill_loss_n):
                tofill[:,t] = eraser(tofill[:,t])
        zeros = th.zeros_like(tofill[:,:,:1,:,:])
        tofill = th.cat([tofill,zeros],-3)

        # -- foward --
        fill = self.forward(tofill,flows)

        # -- report loss --
        # loss = th.mean((clean - fill)**2)
        eps = 1e-3
        loss = th.sqrt(th.mean((clean - fill)**2) + eps**2)
        return fill.detach(),loss

    def training_step_i(self, batch, start, stop):

        # -- unpack batch
        noisy = batch['noisy'][start:stop]/255.
        clean = batch['clean'][start:stop]/255.
        fflow = batch['fflow'][start:stop]
        bflow = batch['bflow'][start:stop]

        # -- add 4-th chnl --
        noisy = self.ensure_chnls(noisy,batch)

        # -- make flow --
        if fflow.shape[-2:] == noisy.shape[-2:]:
            flows = edict({"fflow":fflow,"bflow":bflow})
        else:
            flows = None

        # -- forward --
        deno = self.forward(noisy,flows)

        # -- report loss --
        eps = 1e-3
        loss = th.sqrt(th.mean((clean - deno)**2) + eps**2)
        # loss = th.mean((clean - deno)**2)

        # -- forward fill --
        fill_f = None
        if self.fill_loss:
            fill_f,loss_f = self.training_step_fill(clean,flows)
            loss += self.fill_loss_weight * loss_f

        return deno.detach(),fill_f,clean,loss

    def ensure_chnls(self,noisy,batch):
        if noisy.shape[-3] == self.dd_in:
            return noisy
        elif noisy.shape[-3] == 4 and self.dd_in == 3:
            return noisy[...,:3,:,:].contiguous()
        sigmas = []
        B,t,c,h,w = noisy.shape
        for b in range(B):
            sigma_b = batch['sigma'][b]/255.
            noise_b = th.ones(t,1,h,w,device=sigma_b.device) * sigma_b
            sigmas.append(noise_b)
        sigmas = th.stack(sigmas)
        return th.cat([noisy,sigmas],2)

    def validation_step(self, batch, batch_idx):

        # -- sample noise from simulator --
        self.sample_noisy(batch)

        # -- denoise --
        noisy,clean = batch['noisy']/255.,batch['clean']/255.
        val_index = batch['index'].cpu().item()
        T = noisy.shape[1]

        # -- add 4-th chnl --
        noisy = self.ensure_chnls(noisy,batch)
        # print("val: ",noisy[:,0,-1,0,0],batch['sigma'])

        # -- flow --
        fflow = batch['fflow']
        bflow = batch['bflow']
        if fflow.shape[-2:] == noisy.shape[-2:]:
            flows = edict({"fflow":fflow,"bflow":bflow})
        else:
            flows = None

        # -- forward --
        gpu_mem.print_peak_gpu_stats(False,"val",reset=True)
        with th.no_grad():
            deno = self.forward(noisy,flows)
        mem_res,mem_alloc = gpu_mem.print_peak_gpu_stats(False,"val",reset=True)

        # -- loss --
        loss = th.mean((clean - deno)**2)
        val_psnr = np.mean(compute_psnrs(deno,clean,div=1.)).item()
        val_ssim = np.mean(compute_ssims(deno,clean,div=1.)).item()

        # -- report --
        self.log("val_loss", loss.item(), on_step=False,
                 on_epoch=True,batch_size=1,sync_dist=True)
        self.log("val_mem_res", mem_res, on_step=False,
                 on_epoch=True,batch_size=1,sync_dist=True)
        self.log("val_mem_alloc", mem_alloc, on_step=False,
                 on_epoch=True,batch_size=1,sync_dist=True)
        self.log("val_psnr", val_psnr, on_step=False,
                 on_epoch=True,batch_size=1,sync_dist=True)
        self.log("val_ssim", val_ssim, on_step=False,
                 on_epoch=True,batch_size=1,sync_dist=True)
        self.log("val_index", val_index, on_step=False,
                 on_epoch=True,batch_size=1,sync_dist=True)
        self.log("global_step",self.global_step,on_step=False,
                 on_epoch=True,batch_size=1,sync_dist=True)

        # print("val_psnr: ",val_psnr)
        # print("val_ssim: ",val_ssim)

        # -- channel info --
        # for i in range(deno.shape[2]):
        #     self.log("val_deno_ch_%d_mean" % i,deno[:,:,i].mean().item())
        #     self.log("val_deno_ch_%d_min" % i,deno[:,:,i].min().item())
        #     self.log("val_deno_ch_%d_max" % i,deno[:,:,i].max().item())
        #     self.log("val_noisy_ch_%d_mean" % i,noisy[:,:,i].mean().item())
        #     self.log("val_noisy_ch_%d_min" % i,noisy[:,:,i].min().item())
        #     self.log("val_noisy_ch_%d_max" % i,noisy[:,:,i].max().item())
        #     print("val_deno_ch_%d_mean" % i,deno[:,:,i].mean().item())
        #     print("val_deno_ch_%d_min" % i,deno[:,:,i].min().item())
        #     print("val_deno_ch_%d_max" % i,deno[:,:,i].max().item())
        #     print("val_noisy_ch_%d_mean" % i,noisy[:,:,i].mean().item())
        #     print("val_noisy_ch_%d_min" % i,noisy[:,:,i].min().item())
        #     print("val_noisy_ch_%d_max" % i,noisy[:,:,i].max().item())


        # -- image --
        deno = deno.clamp(0,1)
        noisy = noisy.clamp(0,1)
        if noisy.shape[2] == 4:
            noisy = noisy[:,:,:3]
            deno = deno[:,:,:3]
        # from dev_basics.utils import vid_io
        # vid_io.save_video(noisy,"output/checks","val_noisy_%d"%int(val_index))
        # vid_io.save_video(deno,"output/checks","val_deno_%d"%int(val_index))

        # if not(self.logger is None):
        #     self.logger.log_image(key="val_noisy_"+str(int(val_index)),
        #                           images=[noisy_z[0][t] for t in range(T)])
        #     self.logger.log_image(key="val_deno_"+str(int(val_index)),
        #                           images=[deno[0][t] for t in range(T)])
        self.gen_loger.info("val_psnr: %2.2f" % val_psnr)
        self.gen_loger.info("val_ssim: %.3f" % val_ssim)

    def test_step(self, batch, batch_nb):

        # -- sample noise from simulator --
        self.sample_noisy(batch)

        # -- denoise --
        index = float(batch['index'][0].item())
        noisy,clean = batch['noisy']/255.,batch['clean']/255.
        T = noisy.shape[1]


        # -- add 4-th chnl --
        noisy = self.ensure_chnls(noisy,batch)
        # print("te: ",noisy[:,0,-1,0,0],batch['sigma'])

        # -- flow --
        fflow = batch['fflow']
        bflow = batch['bflow']
        if fflow.shape[-2:] == noisy.shape[-2:]:
            flows = edict({"fflow":fflow,"bflow":bflow})
        else:
            flows = None

        # -- forward --
        gpu_mem.print_peak_gpu_stats(False,"test",reset=True)
        with th.no_grad():
            deno = self.forward(noisy,flows)
        mem_res,mem_alloc = gpu_mem.print_peak_gpu_stats(False,"test",reset=True)

        # -- compare --
        loss = th.mean((clean - deno)**2)
        psnr = np.mean(compute_psnrs(deno,clean,div=1.)).item()
        ssim = np.mean(compute_ssims(deno,clean,div=1.)).item()

        # -- terminal log --
        self.log("test_psnr", psnr, on_step=True, on_epoch=False, batch_size=1)
        self.log("test_ssim", ssim, on_step=True, on_epoch=False, batch_size=1)
        self.log("test_index", index,on_step=True,on_epoch=False,batch_size=1)
        self.log("test_mem_res", mem_res, on_step=True, on_epoch=False, batch_size=1)
        self.log("test_mem_alloc", mem_alloc,on_step=True,on_epoch=False,batch_size=1)
        self.log("global_step",self.global_step,on_step=True,
                 on_epoch=False,batch_size=1)
        # if not(self.logger is None):
        #     self.logger.log_image(key="te_noisy_"+str(int(index)),
        #                           images=[noisy[0][t].clamp(0,1) for t in range(T)])
        #     self.logger.log_image(key="te_deno_"+str(int(index)),
                                  # images=[deno[0][t].clamp(0,1) for t in range(T)])
        self.gen_loger.info("te_psnr: %2.2f" % psnr)
        self.gen_loger.info("te_ssim: %.3f" % ssim)

        # -- log --
        results = edict()
        results.test_loss = loss.item()
        results.test_psnr = psnr
        results.test_ssim = ssim
        results.test_mem_alloc = mem_alloc
        results.test_mem_res = mem_res
        results.test_index = index#.cpu().numpy().item()

        return results

    def uses_spynet(self):
        return hasattr(self.net,"spynet") and not(self.net.spynet is None)

    def num_steps(self) -> int:
        """Get number of steps"""
        # Accessing _data_source is flaky and might break
        if self.nsteps > 0:
            return self.nsteps
        elif self.limit_train_batches > 0:
            dataset_size = self.limit_train_batches
            num_devices = 1
        else:
            dataset = self.trainer.fit_loop._data_source.dataloader()
            dataset_size = len(dataset)
            num_devices = max(1, self.trainer.num_devices)
        acc = self.trainer.accumulate_grad_batches
        num_steps = dataset_size * self.trainer.max_epochs // (acc * num_devices)
        return num_steps

def remove_lightning_load_state(state):
    names = list(state.keys())
    for name in names:
        name_new = name.split(".")[1:]
        name_new = ".".join(name_new)
        state[name_new] = state[name]
        del state[name]
