

def translate_values(field,in_values):
    if field == "attn_mode":
        out_values = []
        for _v in in_values:
            v = translate_attn_mode(_v)
            out_values.append(v)
    elif field == "freeze":
        out_values = []
        for _v in in_values:
            v = translate_freeze(_v)
            out_values.append(v)
    else:
        out_values = [int(v) for v in in_values]
    return out_values

def translate_freeze(_v):
    return _v == "t"

def translate_attn_mode(_v): # keys under "augmented/lewin_ref.py"
    if _v == "pd":
        v = "product_stnls"
    elif _v == "ld":
        v = "l2_stnls"
    elif _v == "wd":
        v = "window_stnls"
    elif _v == "wr":
        v = "window_refactored"
    elif _v == "w":
        v = "window_default"
    else:
        raise ValueError(f"Uknown [attn_mode] type [{_v}]")
    return v

def translate_attn_reset(_v):
    if _v == "f":
        return False
    elif _v == "t":
        return True
    else:
        raise ValueError(f"Uknown [attn_reset] type [{_v}]")

def expand_embed_dim(embed_dim,nblocks=5):
    if isinstance(embed_dim,int):
        exp_embed_dim = [embed_dim for _ in range(nblocks)]
    else:
        exp_embed_dim = embed_dim.split("-")
        exp_embed_dim = [int(v) for v in exp_embed_dim]
    assert len(exp_embed_dim) == nblocks
    return exp_embed_dim

def expand_attn_reset(attn_reset,nblocks=5):
    if isinstance(attn_reset,bool):
        exp_attn_reset = [attn_reset for _ in range(nblocks)]
    else:
        exp_attn_reset = attn_reset.split("-")
        exp_attn_reset = [translate_attn_reset(v) for v in exp_attn_reset]
    assert len(exp_attn_reset) == nblocks
    return exp_attn_reset

def expand_attn_mode(in_attn_mode,nblocks=5):
    if "_" in in_attn_mode:
        attn_modes = [in_attn_mode for _ in range(nblocks)]
    else:
        attn_modes = in_attn_mode.split("-")
        attn_modes = [translate_attn_mode(v) for v in attn_modes]
    assert len(attn_modes) == nblocks
    return attn_modes
