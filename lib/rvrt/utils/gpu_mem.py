import torch as th
import numpy as np

def print_gpu_stats(verbose,name):
    fmt_all = "[%s] Memory Allocated [GB]: %2.3f"
    fmt_res = "[%s] Memory Reserved [GB]: %2.3f"
    th.cuda.empty_cache()
    th.cuda.synchronize()
    mem_alloc = th.cuda.memory_allocated() / 1024**3
    mem_res = th.cuda.memory_reserved() / 1024**3
    if verbose:
        print(fmt_all % (name,mem_alloc))
        print(fmt_res % (name,mem_res))
    return mem_alloc,mem_res

def reset_peak_gpu_stats():
    th.cuda.reset_max_memory_allocated()

def print_peak_gpu_stats(verbose,name,reset=True):
    fmt = "[%s] Peak Memory Allocated [GB]: %2.3f"
    mem_alloc = th.cuda.max_memory_allocated(0) / (1024.**3)
    mem_res = th.cuda.max_memory_reserved(0) / (1024.**3)
    if verbose:
        th.cuda.empty_cache()
        th.cuda.synchronize()
        print(fmt % (name,mem_alloc))
    if reset: th.cuda.reset_peak_memory_stats()
    return mem_alloc,mem_res

class GpuRecord():

    def __init__(self):
        self.mems = [] # reserved
        self.mems_alloc = [] # allocated
        self.names = []

    def sortby(self,names):
        assert set(names) == set(self.names),"must have all keys."
        args = np.argsort(names)
        self.mems = [self.mems[i] for i in args]
        self.mems_alloc = [self.mems_alloc[i] for i in args]
        self.names = [self.names[i] for i in args]

    def __str__(self):
        msg = "--- Gpu Mems ---\n"
        for k,mem_res,mem_alloc in self.items(True):
            msg += "\n%s: %2.3f, %2.3f\n" % (k,mem_res,mem_alloc)
        return msg

    def __getitem__(self,name):
        idx = self.names.index(name)
        gpu_mem = self.mems[idx]
        return gpu_mem

    def items(self,both=False):
        if both:
            return zip(self.names,self.mems,self.mems_alloc)
        else:
            return zip(self.names,self.mems)

    def reset(self):
        print_peak_gpu_stats(False,"",True)

    def snap(self,name):
        mem_alloc,mem_res = print_peak_gpu_stats(False,"",True)
        self.names.append(name)
        self.mems.append(mem_res)
        self.mems_alloc.append(mem_alloc)

