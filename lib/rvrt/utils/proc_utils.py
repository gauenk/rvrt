"""
Processing Utils

"""

import math
import numpy as np
import torch as th
from functools import partial

def _vprint(verbose,*args,**kwargs):
    if verbose:
        print(*args,**kwargs)

def expand2square(timg,factor=16.0):
    t, _, h, w = timg.size()

    X = int(math.ceil(max(h,w)/float(factor))*factor)

    img = th.zeros(t,3,X,X).type_as(timg) # 3, h,w
    mask = th.zeros(t,1,X,X).type_as(timg)

    print(img.size(),mask.size())
    # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
    img[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)] = timg
    mask[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)].fill_(1)

    return img, mask

def get_chunks(size,chunk_size,overlap):
    """

    Thank you to https://github.com/Devyanshu/image-split-with-overlap/

    args:
      size = original size
      chunk_size = size of output chunks
      overlap = percent (from 0.0 - 1.0) of overlap for each chunk

    This code splits an input size into chunks to be used for
    split processing

    """
    points = [0]
    stride = max(int(chunk_size * (1-overlap)),1)
    assert stride > 0
    counter = 1
    while True:
        pt = stride * counter
        if pt + chunk_size >= size:
            points.append(size - chunk_size)
            break
        else:
            points.append(pt)
        counter += 1
    return np.unique(points)

def get_spatial_chunk(vid,h_chunk,w_chunk,size):
    return vid[...,h_chunk:h_chunk+size,w_chunk:w_chunk+size]

def fill_spatial_chunk(vid,ivid,h_chunk,w_chunk,size):
    vid[...,h_chunk:h_chunk+size,w_chunk:w_chunk+size] += ivid

def spatial_chop(ssize,overlap,fwd_fxn,vid,flows=None,verbose=False):
    """
    overlap is a _percent_

    """
    vprint = partial(_vprint,verbose)
    H,W = vid.shape[-2:] # .... H, W
    deno,Z = th.zeros_like(vid),th.zeros_like(vid)
    h_chunks = get_chunks(H,ssize,overlap)
    w_chunks = get_chunks(W,ssize,overlap)
    vprint("h_chunks,w_chunks: ",h_chunks,w_chunks)
    for h_chunk in h_chunks:
        for w_chunk in w_chunks:
            vid_chunk = get_spatial_chunk(vid,h_chunk,w_chunk,ssize)
            vprint("s_chunk: ",h_chunk,w_chunk,vid_chunk.shape)
            if flows: deno_chunk = fwd_fxn(vid_chunk,flows)
            else: deno_chunk = fwd_fxn(vid_chunk)
            ones = th.ones_like(deno_chunk)
            fill_spatial_chunk(deno,deno_chunk,h_chunk,w_chunk,ssize)
            fill_spatial_chunk(Z,ones,h_chunk,w_chunk,ssize)
    deno /= Z # normalize across overlaps
    return deno

def temporal_chop(tsize,overlap,fwd_fxn,vid,flows=None,verbose=False):
    """
    overlap is a __percent__
    """
    vprint = partial(_vprint,verbose)
    nframes = vid.shape[0]
    t_chunks = get_chunks(nframes,tsize,overlap)
    vprint("t_chunks: ",t_chunks)
    deno,Z = th.zeros_like(vid),th.zeros_like(vid)
    for t_chunk in t_chunks:

        # -- extract --
        t_slice = slice(t_chunk,t_chunk+tsize)
        vid_chunk = vid[t_slice]
        vprint("t_chunk: ",t_chunk,vid_chunk.shape)

        # -- process --
        if flows: deno_chunk = fwd_fxn(vid_chunk,flows)
        else: deno_chunk = fwd_fxn(vid_chunk)

        # -- accumulate --
        ones = th.ones_like(deno_chunk)
        deno[t_slice] += deno_chunk
        Z[t_slice] += ones
    deno /= Z
    return deno

