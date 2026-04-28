Tested on Python 3.14.4

Everything was done on my machine locally
M-Series macs run AI very well with some quirks so some hacks were done

get_platform_device(): this function finds which GPU is available and sets the device function to that.


The default backend for torch.compile does not run well with and will crash the compiler

Solution:
'''
backend = "aot_eager" if device.type == "mps" else "inductor"   
model = torch.compile(base_model.to(device), backend=backend)
'''

the repository is not called huggingface/pytorch-transformers and will not work. Huggingface decided to unify everything with transformers.


Some of the packages that will be needed:

- torch
- pandas
- numpy<2; numpy>2 crashes
- matplotlib
- transformers
