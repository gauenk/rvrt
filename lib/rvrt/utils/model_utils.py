import math
import torch as th
import torch.nn as nn
import os
import copy
ccopy = copy.copy
from einops import repeat
from pathlib import Path
from collections import OrderedDict

from .model_keys import translate_attn_mode,expand_attn_mode
from .qkv_convert import qkv_convert_state,block_name2num

def freeze(model):
    for p in model.parameters():
        p.requires_grad=False

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad=True

def is_frozen(model):
    x = [p.requires_grad for p in model.parameters()]
    return not all(x)

def save_checkpoint(model_dir, state, session):
    epoch = state['epoch']
    mpath = "model_epoch_{}_{}.pth".format(epoch,session)
    model_out_path = os.path.join(model_dir,mpath)
    th.save(state, model_out_path)


def reset_mismatch(model,state_dict):
    model_state_dict = model.state_dict()
    for k in state_dict:
        if k in model_state_dict:
            s = model_state_dict[k]
            # if s.dtype != th.long:
            #     state_dict[k] = th.randn_like(s).clip(-1,1)
            if state_dict[k].shape != model_state_dict[k].shape:
                s = model_state_dict[k]
                state_dict[k] = th.randn_like(s).clip(-1,1)

def load_checkpoint_qkv(model, weights,in_attn_modes, out_attn_modes,
                        embed_dim,prefix="module.",reset_new=False,
                        attn_reset=False,strict=True,skip_mismatch_model_load=False):
    if weights is None: return
    checkpoint = th.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = qkv_convert_state(
        state_dict,in_attn_modes,out_attn_modes,embed_dim,
        prefix=prefix,reset_new=reset_new,attn_reset=attn_reset)
    if skip_mismatch_model_load:
        reset_mismatch(model,new_state_dict)
    model.load_state_dict(new_state_dict,strict=strict)

def load_checkpoint_module(model, weights):
    checkpoint = th.load(weights)
    try:
        # model.load_state_dict(checkpoint["state_dict"])
        raise ValueError("")
    except Exception as e:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

def load_checkpoint_multigpu(model, weights):
    checkpoint = th.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

def load_start_epoch(weights):
    checkpoint = th.load(weights)
    epoch = checkpoint["epoch"]
    return epoch

def load_optim(optimizer, weights):
    checkpoint = th.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer'])
    for p in optimizer.param_groups: lr = p['lr']
    return lr

def get_arch(opt):
    from model import UNet,Uformer,Uformer_Cross,Uformer_CatCross
    arch = opt.arch

    print('You choose '+arch+'...')
    if arch == 'UNet':
        model_restoration = UNet(dim=opt.embed_dim)
    elif arch == 'Uformer':
        model_restoration = Uformer(img_size=opt.train_ps,embed_dim=opt.embed_dim,win_size=opt.win_size,token_projection=opt.token_projection,token_mlp=opt.token_mlp)
    elif arch == 'Uformer16':
        model_restoration = Uformer(img_size=opt.train_ps,embed_dim=16,win_size=8,token_projection='linear',token_mlp='leff')
    elif arch == 'Uformer32':
        model_restoration = Uformer(img_size=opt.train_ps,embed_dim=32,win_size=8,token_projection='linear',token_mlp='leff')
    elif arch == 'Uformer_CatCross':
        model_restoration = Uformer_CatCross(img_size=opt.train_ps,embed_dim=opt.embed_dim,win_size=8,token_projection=opt.token_projection,token_mlp=opt.token_mlp)
    elif arch == 'Uformer_Cross':
        model_restoration = Uformer_Cross(img_size=opt.train_ps,embed_dim=opt.embed_dim,win_size=opt.win_size,token_projection=opt.token_projection,token_mlp=opt.token_mlp)
    else:
        raise Exception("Arch error!")

    return model_restoration

def remove_lightning_load_state(state):
    names = list(state.keys())
    for name in names:
        name_new = name.split(".")[1:]
        name_new = ".".join(name_new)
        state[name_new] = state[name]
        del state[name]


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#      Loading Checkpoint by Filename of Substr
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


def load_checkpoint(model,use_train,substr="",croot="output/checkpoints/"):
    # -- do we load --
    load = use_train == "true" or use_train is True

    # -- load to model --
    if load:
        mpath = get_recent_filename(croot,substr)
        print("Loading Model Checkpoint: ",mpath)
        state = th.load(mpath)['state_dict']
        remove_lightning_load_state(state)
        model.load_state_dict(state)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#     Loading the Most Recent File
#
#     think:
#     "path/to/dir/","this-is-my-uuid"
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def get_recent_filename(root,substr):
    root = Path(root)
    if not root.exists():
        raise ValueError(f"Load directory [{root}] does not exist.")
    files = []
    for fn in root.iterdir():
        fn = str(fn)
        if substr == "" or substr in fn:
            files.append(fn)
    files = sorted(files,key=os.path.getmtime)
    files = [f for f in reversed(files)]

    # -- error --
    if len(files) == 0:
        raise ValueError(f"Unable to file any files with substr [{substr}]")

    return str(files[0])

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#     Modifying Layers In-Place after Loading
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def reset_product_attn_mods(model):
    for name,param in model.named_parameters():
        if "relative_position_bias_table" in name:
            submod = model
            submods = name.split(".")
            for submod_i in submods:
                submod = getattr(submod,submod_i)
            submod.data = th.randn_like(submod.data).clamp(-1,1)/100.

def filter_rel_pos(model,in_attn_mode):
    attn_modes = expand_attn_mode(in_attn_mode)
    for name,param in model.named_parameters():
        if "relative_position_bias_table" in name:
            bname = name.split(".")[0]
            bnum = block_name2num(bname)
            attn_mode_b = attn_modes[bnum]
            if attn_mode_b == "product_stnls":
                submod = model
                submods = name.split(".")
                for submod_i in submods[:-1]:
                    submod = getattr(submod,submod_i)
                setattr(submod,submods[-1],None)

def get_product_attn_params(model):
    params = []
    for name,param in model.named_parameters():
        if "relative_position_bias_table" in name:
            # delattr(model,name)
            param.requires_grad_(False)
            continue
        params.append(param)
    return params

def apply_freeze(model,freeze):
    if freeze is False: return
    unset_names = []
    for name,param in model.named_parameters():
        # print(name)
        bname = name.split(".")[0]
        bnum = block_name2num(bname)
        if bnum == -1: unset_names.append(name)
        freeze_b = freeze[bnum]
        if freeze_b is True:
            param.requires_grad_(False)
    # print(unset_names)

# -- embed dims --
def expand_embed_dims(attn_modes,embed_dim_w,embed_dim_pd):
    exp_embed_dims = []
    for attn_mode in attn_modes:
        # print("attn_mode: ",attn_mode)
        if "window" in attn_mode:
            exp_embed_dims.append(embed_dim_w)
        elif "product" in attn_mode:
            exp_embed_dims.append(embed_dim_pd)
        else:
            raise ValueError(f"Uknown attn_mode [{attn_mode}]")
    assert len(exp_embed_dims) == 5
    return exp_embed_dims
