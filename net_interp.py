import sys
import torch
from collections import OrderedDict

alpha = float(sys.argv[1])
models_path = sys.argv[2]

net_PSNR_path = f"{models_path}RRDB_PSNR_x4.pth"
net_ESRGAN_path = f"{models_path}RRDB_ESRGAN_x4.pth"
net_interp_path = f"{models_path}interp_{int(alpha*10)}.pth"

net_PSNR = torch.load(net_PSNR_path)
net_ESRGAN = torch.load(net_ESRGAN_path)
net_interp = OrderedDict()

print('Interpolating with alpha = ', alpha)

for k, v_PSNR in net_PSNR.items():
    v_ESRGAN = net_ESRGAN[k]
    net_interp[k] = (1 - alpha) * v_PSNR + alpha * v_ESRGAN

torch.save(net_interp, net_interp_path)
