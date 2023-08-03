
import math,pickle
import numpy as np
import torch as th
from einops import rearrange

def optional(pydict,key,default):
    if pydict is None: return default
    elif key in pydict: return pydict[key]
    else: return default

def optional_delete(pydict,key):
    if pydict is None: return
    elif key in pydict: del pydict[key]
    else: return

def tuple_as_int(elem):
    if hasattr(elem,"__getitem__"):
        return elem[0]
    else:
        return elem

def task_keys(task):
    if "blur" in task:
        return "blur","sharp"
    else:
        return "noisy","clean"

def assert_nonan(tensor):
    assert th.any(th.isnan(tensor)).item() is False

def rslice_pair(vid_a,vid_b,coords):
    vid_a = rslice(vid_a,coords)
    vid_b = rslice(vid_b,coords)
    return vid_a,vid_b

def rslice(vid,coords):
    if coords is None: return vid
    if len(coords) == 0: return vid
    if th.is_tensor(coords):
        coords = coords.type(th.int)
        coords = list(coords.cpu().numpy())
    fs,fe,t,l,b,r = coords
    return vid[fs:fe,:,t:b,l:r]

def write_pickle(fn,obj):
    with open(str(fn),"wb") as f:
        pickle.dump(obj,f)

def read_pickle(fn):
    with open(str(fn),"rb") as f:
        obj = pickle.load(f)
    return obj

def full2blocks(img,blocks):
    one,c,h,w = img.shape
    assert one == 1
    bh,bw = 256,256
    nblocks = blocks.shape[0]
    img_blocks = th.zeros((nblocks,c,bh,bw),device=img.device)
    for b in range(nblocks):
        sh,sw = blocks[b][0],blocks[b][1]
        eh,ew = sh+bh,sw+bw
        slice_h = slice(sh,eh)
        slice_w = slice(sw,ew)
        img_blocks[b] = img[:,:,slice_h,slice_w]
    return img_blocks

def stacked2full(img,blocks):
    r,c,h,w = img.shape
    rh = r//8
    rw = r//rh
    shape_str = '(rh rw) c h w -> 1 c (rh h) (rw w)'
    # img = rearrange(img,shape_str,rh=rh,rw=rw)

    # -- blocks for indexing --
    print(blocks)
    # blocks = get_blocks()
    print(blocks)
    # blocks[:,0] -= blocks[:,0].min()
    # blocks[:,1] -= blocks[:,1].min()
    print(blocks[:,0].max(),blocks[:,0].min())
    print(blocks[:,1].max(),blocks[:,1].min())
    print(blocks.shape)
    nblocks = blocks.shape[0]

    # -- new img --
    nh,nw = blocks[:,0].max()+256,blocks[:,1].max()+256
    img_new = th.zeros(1,c,nh,nw)

    for b in range(nblocks):
        sh,sw = blocks[b][0],blocks[b][1]
        eh,ew = sh+h,sw+w
        print(sh,sw)
        img_new[:,:,sh:eh,sw:ew] = img[b]
    return img_new


# def get_blocks():
#     blocks = [[1601, 2413],[1601, 2669],[1857, 2413],[1857, 2669],[1063,   57],
#               [1063,  313],[1319,   57],[1319,  313],[ 491,  329],[ 491,  585],
#               [ 747,  329],[ 747,  585],[2203, 1705],[2203, 1961],[2459, 1705],
#               [2459, 1961],[ 699, 3149],[ 699, 3405],[ 955, 3149],[ 955, 3405],
#               [1583, 1467],[1583, 1723],[1839, 1467],[1839, 1723],[ 241, 2087],
#               [ 241, 2343],[ 497, 2087],[ 497, 2343],[2337, 1093],[2337, 1349],
#               [2593, 1093],[2593, 1349]]
#     blocks = np.array(blocks)-1
#     return blocks

def tiles2img(img_tiles,nh,nw,H,W):
    ntiles,c,tile_h,tile_w = img_tiles.shape
    img = th.zeros((1,c,H,W),device=img_tiles.device)
    Z = th.zeros((1,c,H,W),device=img_tiles.device)
    ones = th.ones_like(img_tiles[0])
    tidx = 0
    for hi in range(nh):
        for wi in range(nw):
            start_h = hi * tile_h
            end_h = (hi+1) * tile_h
            if end_h >= H:
                end_h = H - 1
                start_h = end_h - tile_h
            slice_h = slice(start_h,end_h)

            start_w = wi * tile_w
            end_w = (wi+1) * tile_w
            if end_w >= W:
                end_w = W - 1
                start_w = end_w - tile_w
            slice_w = slice(start_w,end_w)

            # -- assign --
            img[...,slice_h,slice_w] += img_tiles[tidx]
            Z[...,slice_h,slice_w] += ones
            tidx += 1
    img /= (Z+1e-10)
    return img

def img2tiles(img,tile_h,tile_w):

    # -- unpack --
    one,c,H,W = img.shape
    assert one == 1
    imgs = []

    # -- each section --
    nh = (H - 1)//tile_h + 1
    nw = (W - 1)//tile_w + 1
    print(img.shape)

    # -- iterate --
    for hi in range(nh):
        for wi in range(nw):
            start_h = hi * tile_h
            end_h = (hi+1) * tile_h
            if end_h >= H:
                end_h = H - 1
                start_h = end_h - tile_h
            slice_h = slice(start_h,end_h)

            start_w = wi * tile_w
            end_w = (wi+1) * tile_w
            if end_w >= W:
                end_w = W - 1
                start_w = end_w - tile_w
            slice_w = slice(start_w,end_w)

            img_i = img[...,slice_h,slice_w]
            imgs.append(img_i)
    imgs = th.cat(imgs)
    return imgs,nh,nw

def expand2square(timg,factor=16.0):
    _, _, h, w = timg.size()

    X = int(math.ceil(max(h,w)/float(factor))*factor)

    img = th.zeros(1,3,X,X).type_as(timg) # 3, h,w
    mask = th.zeros(1,1,X,X).type_as(timg)

    # print(img.size(),mask.size())
    # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
    img[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)] = timg
    mask[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)].fill_(1)

    return img, mask.bool()

