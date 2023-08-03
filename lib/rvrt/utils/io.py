# -- linalg --
import torch as th
import numpy as np
from einops import rearrange

# -- file io --
from PIL import Image
from pathlib import Path

def save_burst(burst,root,name,fstart=0,div=None,fmt="image"):

    # -- path --
    root = Path(str(root))
    if not root.exists():
        print(f"Making dir for save_burst [{str(root)}]")
        root.mkdir(parents=True)
    assert root.exists()
    ext = get_ext(fmt)

    # -- save --
    save_fns = []
    nframes = burst.shape[0]
    for t in range(nframes):
        fid = t + fstart
        img_t = burst[t]
        path_t = root / ("%s_%05d.%s" % (name,fid,ext))
        save_image(img_t,str(path_t),div,fmt)
        save_fns.append(str(path_t))
    return save_fns

def get_ext(fmt):
    if fmt in ["image","png"]:
        ext = "png"
    elif fmt == "np":
        ext = "npy"
    else:
        raise ValueError(f"Uknown save format [{fmt}]")
    return ext

def save_image(image,path,div=None,fmt="image"):

    # -- to numpy --
    if th.is_tensor(image):
        image = image.detach().cpu().numpy()

    # -- only rescale if not specified --
    no_div = div is None

    # -- scale with div --
    if not(div is None):
        image /= div

    # -- rescale from fold --
    if image.max() > 500 and no_div: # probably from a fold
        image /= image.max()

    # -- normalize if [0,1] --
    if image.max() < 100 and no_div:
        image = image*255.

    if fmt in ["image","png"]:
        # -- to uint8 --
        image = np.clip(image,0,255).astype(np.uint8)

        # -- remove single color --
        image = rearrange(image,'c h w -> h w c')
        image = image.squeeze()

        # -- save --
        img = Image.fromarray(image)
        img.save(path)
    elif fmt == "np":
        np.save(path,image)
    else:
        raise ValueError(f"Uknown save format [{fmt}]")
