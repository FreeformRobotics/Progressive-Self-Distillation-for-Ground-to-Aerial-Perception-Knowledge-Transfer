from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from DataLoader import Airsim
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='datasets/airsim/',help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='airsim', help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=9,help="num classes (default: None)")
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


def get_dataset(opts, path='test/', csv_file='uav_test2.csv'):
    train_transform = et.ExtCompose([
        et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    val_transform = et.ExtCompose([
        #et.ExtResize( 512 ),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])


    val_dst = Airsim(img_dir=opts.data_root + path, csv_file = opts.data_root + csv_file, transform=val_transform)
    return val_dst


def validate(opts, model, loader, device, metrics, ret_samples_ids=None, path=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if not os.path.exists(path):
        os.mkdir(path)

    with torch.no_grad():
        for i, (images, labels, img_name, label_name) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])

                colors = ['lime', 'gray', 'bisque', 'blue','yellow', 'pink', 'white', 'brown', 'cyan']
                vmax = preds.max() + 1
                cmap = matplotlib.colors.ListedColormap(colors[:vmax])

                matplotlib.pyplot.imsave(path + str(i) + '.png',preds.squeeze(),cmap=cmap)

        score = metrics.get_results()
    return score, ret_samples


def main():
    opts = get_argparser().parse_args()

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
    checkpoint = torch.load('./runs/airsim.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])

    model = nn.DataParallel(model)
    model.to(device)

    metrics = StreamSegMetrics(opts.num_classes)

    if opts.test_only:
        model.eval()
        vis_sample_id = None
            # Setup dataloader
        val_dst = get_dataset(opts,path='test/')
        val_loader = data.DataLoader(val_dst, batch_size=opts.val_batch_size, shuffle=False, num_workers=2)

        ########## generalization performance

        if not os.path.exists('results_airsim'):
            os.mkdir('results_airsim')

        val_score, ret_samples = validate(opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id, path='results_airsim/uav_test/')
        print(metrics.to_str(val_score))


        return


if __name__ == '__main__':
    main()
