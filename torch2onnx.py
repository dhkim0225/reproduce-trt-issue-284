import argparse
import os

import numpy as np
import onnx
import torch
import torch.nn as nn


class TinyModel(nn.Module):
    def __init__(self, upsample_mode):
        super().__init__()
        self.expander = nn.Conv2d(3, 192, 1, 1)
        upsamples = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                     nn.Upsample(scale_factor=2, mode='nearest'),
                     nn.Upsample((256, 256), mode='bilinear', align_corners=False),
                     nn.Upsample((256, 256), mode='bilinear', align_corners=True),
                     nn.Upsample((256, 256), mode='nearest')]

        self.upsample = upsamples[upsample_mode]

    def forward(self, x):
        x = self.expander(x)
        x = self.upsample(x)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save_folder', type=str, required=True)
    args = parser.parse_args()
    if not os.path.isdir(args.save_folder):
        os.mkdir(args.save_folder)

    device = torch.device('cuda:0')
    sample_input = torch.rand(1, 3, 128, 128).to(device)
    for i in range(6):
        model = TinyModel(upsample_mode=i).to(device)
        model.eval()

        # export onnx file
        onnx_path = os.path.join(args.save_folder, '{}.onnx'.format(i))
        torch.onnx.export(model, sample_input, onnx_path,
                          input_names=['input_img'],
                          output_names=['output'],
                          opset_version=11)

        # save output
        sample_out_path = os.path.join(args.save_folder, str(i))
        sample_output = model(sample_input)
        np.save(sample_out_path + '_inp.npy', sample_input.data.cpu().numpy())
        np.save(sample_out_path + '_out.npy', sample_output.data.cpu().numpy())

        # check valid is graph or not
        try:
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_graph(onnx_model)
        except:
            print('{} is invalid_graph'.format(onnx_path))
