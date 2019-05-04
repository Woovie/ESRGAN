import sys
import os.path
import glob
import cv2
import numpy as np
import torch
import architecture as arch

model_path = sys.argv[1]
source_path = sys.argv[2]
destination_path = sys.argv[3]
device = torch.device('cuda')
baseFileName = os.path.splitext(os.path.basename(source_path))[0]

if (not os.path.exists(destination_path)):
    os.mkdir(destination_path)

for modelFile in glob.glob(f"{model_path}/*"):
    modelName = os.path.splitext(os.path.basename(modelFile))[0]
    model = arch.RRDB_Net(3, 3, 64, 23, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', mode='CNA', res_scale=1, upsample_mode='upconv')
    model.load_state_dict(torch.load(modelFile), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    img = cv2.imread(source_path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)
    output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite(f"{destination_path}/{baseFileName}_{modelName}.png", output)