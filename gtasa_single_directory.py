import sys
import os.path
import glob
import cv2
import numpy as np
import torch
import architecture as arch

model_path = sys.argv[1]
source_path = sys.argv[2]
device = torch.device('cuda')
model = arch.RRDB_Net(3, 3, 64, 23, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', mode='CNA', res_scale=1, upsample_mode='upconv')
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
for k, v in model.named_parameters():
    v.requires_grad = False
model = model.to(device)

def init():
    print(f"GTASA PNG Upscaler ESRGAN Model\nModel file: {model_path}\nSource files: {source_path}")
    if os.path.isdir(source_path):
        for imageFile in glob.glob(f"{source_path}/*"):
            destinationImageFile = imageFile.replace('raw', 'converted')
            modifyImage (imageFile, destinationImageFile)

def modifyImage(imagePath, destinationPath):
    print(f"Modifying {imagePath}...")
    modifiedImage = processImage (imagePath)
    cv2.imwrite(destinationPath, modifiedImage)
    print(f"Saved to {destinationPath}")

def processImage(imagePath):
    img = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)
    output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    return output

init()