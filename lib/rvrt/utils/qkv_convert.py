
import torch as th
import copy
ccopy = copy.copy
from collections import OrderedDict
from .model_keys import translate_attn_mode,expand_attn_mode
from .model_keys import expand_attn_reset,expand_embed_dim

def qkv_convert_lin2conv(new_state_dict,name,val):
    if "to_q" in name:
        if "weight" in name:
            if val.data.ndim < 4:
                val.data = val.data[:,:,None,None]
            new_state_dict[name] = val.data
        elif "bias" in name:
            new_state_dict[name] = val.data
    elif "to_kv" in name:
        if "weight" in name:
            # -- shapes --
            half = val.shape[0]//2
            if val.data.ndim < 4:
                val.data = val.data[:,:,None,None]

            # -- create v --
            name_v = ccopy(name)
            name_v = name_v.replace("to_kv","to_k")
            new_state_dict[name_v] = val.data[:half,:]

            # -- create k --
            name_k = ccopy(name)
            name_k = name_k.replace("to_kv","to_v")
            new_state_dict[name_k] = val.data[half:,:]

        if "bias" in name:
            # -- shapes --
            half = val.shape[0]//2

            # -- create v --
            name_v = ccopy(name)
            name_v = name_v.replace("to_kv","to_k")
            new_state_dict[name_v] = val.data[:half,...]

            # -- create k --
            name_k = ccopy(name)
            name_k = name_k.replace("to_kv","to_v")
            new_state_dict[name_k] = val.data[half:,...]
    else: # already converted.
        new_state_dict[name] = val
        print("[WARNING] What??: ",name,val.shape)

def qkv_convert_conv2lin(new_state_dict,name,val):
    to_kv = "to_k" in name
    to_kv = to_kv or ("to_v" in name)
    if "to_q" in name:
        if "weight" in name:
            if val.data.ndim == 4:
                val.data = val.data[:,:,0,0]
            new_state_dict[name] = val.data
        elif "bias" in name:
            new_state_dict[name] = val.data
    elif to_kv:
        if "weight" in name:

            # -- shapes --
            half = val.shape[0]
            if val.data.ndim == 4:
                val.data = val.data[:,:,0,0]

            # -- specify kv --
            name_kv = ccopy(name)
            if "to_k" in name_kv:
                slice_half = slice(None,half,None)
                name_kv = name_kv.replace("to_k","to_kv")
            elif "to_v" in name_kv:
                slice_half = slice(half,None,None)
                name_kv = name_kv.replace("to_v","to_kv")
            else:
                raise ValueError("")

            # -- create kv --
            # print(name,name_kv,val.data.shape)
            if name_kv in new_state_dict:
                new_state_dict[name_kv][slice_half] = val.data
            else:
                shell = th.zeros_like(val.data).repeat(2,1)
                new_state_dict[name_kv] = shell
                new_state_dict[name_kv][slice_half] = val.data

        if "bias" in name:
            # -- shapes --
            half = val.shape[0]

            # -- specify kv --
            name_kv = ccopy(name)
            if "to_k" in name_kv:
                slice_half = slice(None,half,None)
                name_kv = name_kv.replace("to_k","to_kv")
            elif "to_v" in name_kv:
                slice_half = slice(half,None,None)
                name_kv = name_kv.replace("to_v","to_kv")
            else:
                raise ValueError("")

            # -- create kv --
            if name_kv in new_state_dict:
                new_state_dict[name_kv][slice_half] = val.data
            else:
                shell = th.zeros_like(val.data).repeat(2)
                new_state_dict[name_kv] = shell
                new_state_dict[name_kv][slice_half] = val.data

def block_name2num(name):

    # -- how to compute id --
    fields = ["encoderlayer","decoderlayer",
              "dowsample","upsample"]
    use_split = False
    for field in fields:
        use_split = use_split or field in name

    # -- compute id --
    if use_split:
        encdec,encdec_id = name.split("_")
        if encdec == "encoderlayer":
            return int(encdec_id)
        elif encdec == "decoderlayer":
            return 3 - int(encdec_id)
        elif encdec == "dowsample":
            return int(encdec_id)
        elif encdec == "upsample":
            return int(encdec_id)
        else:
            raise ValueError(f"Uknown name [{name}]")
    elif name == "conv":
        return 4
    else:
        return -1
        # raise ValueError(f"Uknown name [{name}]")

def get_attn_mode_cat(attn_mode):
    if "window" in attn_mode:
        return "window"
    elif "product" in attn_mode:
        return "product"
    else:
        raise ValueError(f"Uknown attention mode [{attn_mode}]")

def qkv_convert_state(state_dict,in_attn_modes,out_attn_modes,
                      embed_dim,prefix="module.",keep=False,
                      reset_new=False,attn_reset=False):

    # -- io attn modes --
    in_attn_modes = expand_attn_mode(in_attn_modes)
    out_attn_modes = expand_attn_mode(out_attn_modes)
    attn_reset = expand_attn_reset(attn_reset)
    # embed_dim = expand_embed_dim(embed_dim)
    # nheads = [2**l for l in range(5)]

    # -- init --
    nskip = len(prefix)
    new_state_dict = OrderedDict()

    for k, val in state_dict.items():

        # -- standard mod --
        name = k[nskip:] if prefix in k else k
        if keep: name_s = prefix + name
        else: name_s = name

        # -- skip non attn modules --
        if not("attn.qkv" in name):
            new_state_dict[name_s] = val.data

        # -- mod qkv --
        if "attn.qkv" in name:

            # -- get block id --
            block_name = name.split(".")[0]
            l = block_name2num(block_name)

            # -- extract modes --
            in_mode = get_attn_mode_cat(in_attn_modes[l])
            out_mode = get_attn_mode_cat(out_attn_modes[l])
            # print(l,in_mode,out_mode)
            no_match = in_mode != out_mode

            # # -- correct dimension --
            mod_shape = False
            # ndim = val.data.ndim
            # if ndim == 2:
            #     nftrs = val.data.shape[1]
            #     nftrs_per_head = nftrs // nheads[l]
            #     mod_shape = not(nftrs_per_head == embed_dim[l])
            # if mod_shape:
            #     new_size = embed_dim[l] * nheads[l]
            #     val.data = val.data[:,:new_size]

            # -- reset values if not equal --
            reset_it = reset_new and no_match or mod_shape
            if reset_new and no_match:
                val.data[...] = th.randn_like(val.data).clip(-1,1)

            # -- reset values if attn reset true --
            if attn_reset[l]:
                val.data[...] = th.randn_like(val.data).clip(-1,1)

            # -- if matching --
            if in_mode == out_mode:
                new_state_dict[name_s] = val
            elif out_mode == "product" and in_mode == "window":
                qkv_convert_lin2conv(new_state_dict,name_s,val)
            elif out_mode == "window" and in_mode == "product":
                qkv_convert_conv2lin(new_state_dict,name_s,val)
            else:
                raise NotImplementedError("in/out mode should be caught already.")

    return new_state_dict
