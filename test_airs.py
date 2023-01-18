from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
from torch.nn import functional as F
from torch.utils import data
from DataLoader import Airs
from utils import ext_transforms as et
from utils import img_transforms as et_img
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
import cv2
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--epochs", type=str, default=500)
    parser.add_argument("--data_root", type=str, default='datasets/airs/',help="path to Dataset")
    parser.add_argument("--num_classes", type=int, default=13, help="num classes (default: None)")
    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet101',
                        choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=True)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results_airsim\"")

    parser.add_argument("--val_batch_size", type=int, default=1, help='batch size for validation (default: 4)')
    parser.add_argument("--gpu_id", type=str, default='0',help="GPU ID")
    return parser


colors = ["#804080",
          "#F423E8",
          "#DC143C",
          "#0000E8",
          "#770B20",
          "#464646",
          "#66669C",
          "#BE9999",
          "#DCDC00",
          "#FAAA1E",
          "#6B8E23",
          "#4682B4",
          "#000000"]


def get_dataset(opts, file):
    """ Dataset And Augmentation
    """
    train_transform = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    data_dst = Airs(img_dir=opts.data_root, csv_file=opts.data_root + file, transform=train_transform)

    return data_dst




def validate(opts, model, loader, device, metrics, ret_samples_ids=None, path=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    if not os.path.exists(path):
        os.mkdir(path)

    with torch.no_grad():
        for i, (images, labels, img_name, label_name) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            outputs = F.interpolate(outputs, size=[images.size(
                2), images.size(3)], mode='bilinear', align_corners=False)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            vmax = preds.max() + 1
            cmap = matplotlib.colors.ListedColormap(colors[:vmax])
            matplotlib.pyplot.imsave(path  + img_name[0][-10:], preds.squeeze(), cmap=cmap)

            # cv2.imwrite(path  + img_name[0][-10:], preds.squeeze(),preds)

            vmax = labels.max() + 1
            cmap = matplotlib.colors.ListedColormap(colors[:vmax])
            matplotlib.pyplot.imsave(path  + img_name[0][-10:], targets.squeeze(), cmap=cmap)

        score = metrics.get_results()
    return score, ret_samples




opts = get_argparser().parse_args()


def main():

    opts.num_classes = 13
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Set up model
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }

    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    checkpoint = torch.load('./runs/airs.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])

    model = nn.DataParallel(model)
    model.to(device)

    metrics = StreamSegMetrics(opts.num_classes)
    if opts.test_only:
        model.eval()
        vis_sample_id = None

        if not os.path.exists('results_airs'):
            os.mkdir('results_airs')

        val_dst = get_dataset(opts, file='uav02_test_gt.csv')
        val_loader = data.DataLoader(
            val_dst, batch_size=1, shuffle=True, num_workers=4)
        val_score, ret_samples = validate(opts=opts, model=model, loader=val_loader, device=device,
                                          metrics=metrics, ret_samples_ids=vis_sample_id, path='results_airs/uav02/')
        print(metrics.to_str(val_score))

        val_dst = get_dataset(opts, file='uav03_test_gt.csv')
        val_loader = data.DataLoader(
            val_dst, batch_size=1, shuffle=True, num_workers=4)
        val_score, ret_samples = validate(opts=opts, model=model, loader=val_loader, device=device,
                                          metrics=metrics, ret_samples_ids=vis_sample_id, path='results_airs/uav03/')
        print(metrics.to_str(val_score))

        val_dst = get_dataset(opts, file='uav04_test_gt.csv')
        val_loader = data.DataLoader(
            val_dst, batch_size=1, shuffle=True, num_workers=4)
        val_score, ret_samples = validate(opts=opts, model=model, loader=val_loader, device=device,
                                          metrics=metrics, ret_samples_ids=vis_sample_id, path='results_airs/uav04/')
        print(metrics.to_str(val_score))

        val_dst = get_dataset(opts, file='uav05_test_gt.csv')
        val_loader = data.DataLoader(
            val_dst, batch_size=1, shuffle=True, num_workers=4)
        val_score, ret_samples = validate(opts=opts, model=model, loader=val_loader, device=device,
                                          metrics=metrics, ret_samples_ids=vis_sample_id, path='results_airs/uav05/')
        print(metrics.to_str(val_score))

        val_dst = get_dataset(opts, file='uav06_test_gt.csv')
        val_loader = data.DataLoader(
            val_dst, batch_size=1, shuffle=True, num_workers=4)
        val_score, ret_samples = validate(opts=opts, model=model, loader=val_loader, device=device,
                                          metrics=metrics, ret_samples_ids=vis_sample_id, path='results_airs/uav06/')
        print(metrics.to_str(val_score))

        val_dst = get_dataset(opts, file='uav07_test_gt.csv')
        val_loader = data.DataLoader(
            val_dst, batch_size=1, shuffle=True, num_workers=4)
        val_score, ret_samples = validate(opts=opts, model=model, loader=val_loader, device=device,
                                          metrics=metrics, ret_samples_ids=vis_sample_id, path='results_airs/uav07/')
        print(metrics.to_str(val_score))

        val_dst = get_dataset(opts, file='uav08_test_gt.csv')
        val_loader = data.DataLoader(
            val_dst, batch_size=1, shuffle=True, num_workers=4)
        val_score, ret_samples = validate(opts=opts, model=model, loader=val_loader, device=device,
                                          metrics=metrics, ret_samples_ids=vis_sample_id, path='results_airs/uav08/')
        print(metrics.to_str(val_score))

        val_dst = get_dataset(opts, file='uav09_test_gt.csv')
        val_loader = data.DataLoader(
            val_dst, batch_size=1, shuffle=True, num_workers=4)
        val_score, ret_samples = validate(opts=opts, model=model, loader=val_loader, device=device,
                                          metrics=metrics, ret_samples_ids=vis_sample_id, path='results_airs/uav09/')
        print(metrics.to_str(val_score))
     

        return


if __name__ == '__main__':
    main()
