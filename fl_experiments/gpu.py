import os
import GPUtil

def get_available_gpu_ids(numGPUs=1, debug=False):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    device_ids = GPUtil.getAvailable(limit=numGPUs, maxLoad=0.5, maxMemory=0.5)

    GPUstring = " ".join(str(x) for x in device_ids)
    GPUstring = GPUstring.replace(" ", ",")
    if debug:
        print(GPUstring)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUstring)

    return device_ids