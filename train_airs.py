from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
from torch.nn import functional as F
from torch.utils import data
from DataLoader import Airs,Airs_img
from utils import ext_transforms as et
from utils import img_transforms as et_img
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
# from utils.visualizer import Visualizer
import transformmasks
import cv2
import copy
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import pdb


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--epochs", type=str, default=500)
    parser.add_argument("--data_root", type=str, default='datasets/airs/',
                        help="path to Dataset")
    parser.add_argument("--num_classes", type=int, default=13,
                        help="num classes (default: None)")
    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet101',
                        choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int,
                        default=16, choices=[8, 16])
    # Train Options
    parser.add_argument("--total_itrs", type=int, default=5000,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=2500)

    parser.add_argument("--batch_size", type=int, default=8,
                        help='batch size (default: 16)')
 
    parser.add_argument("--crop_size", type=int, default=513)
    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training",
                        action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='focal_loss',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")
    parser.add_argument('--name', default='airs', type=str,
                        help='name of experiment')

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    parser.add_argument('--alpha', default=0.75, type=float)
    parser.add_argument('--lambda-u', default=1, type=float)
    parser.add_argument('--T', default=0.5, type=float)
    parser.add_argument('--ema-decay', default=0.999, type=float)

    # # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    # parser.add_argument("--vis_port", type=str, default=None,
    #                     help='port for visdom')
    # parser.add_argument("--vis_env", type=str, default='main',
    #                     help='env for visdom')
    # parser.add_argument("--vis_num_samples", type=int, default=8,
    # help='number of samples for visualization (default: 8)')
    return parser

def get_dataset_training(opts, train_file):
    """ Dataset And Augmentation
    """
    train_transform = et.ExtCompose([
        et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    data_dst = Airs(img_dir=opts.data_root, csv_file=opts.data_root + train_file, transform=train_transform)

    return data_dst


def get_dataset_unlabel(opts, test_file='uav01.csv'):
    """ Dataset And Augmentation
    """

    train_transform = et_img.ExtCompose([
        # et.ExtResize( 512 ),
        # et_img.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
        et_img.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        et_img.ExtRandomHorizontalFlip(),
        et_img.ExtToTensor(),
        et_img.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    data_dst = Airs_img(img_dir=opts.data_root, csv_file=opts.data_root + test_file, transform=train_transform)

    return data_dst


def get_dataset_unlabel2(opts, test_file):
    """ Dataset And Augmentation
    """

    train_transform = et_img.ExtCompose([
        et_img.ExtToTensor(),
        et_img.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    data_dst = Airs_img(img_dir=opts.data_root, csv_file=opts.data_root + test_file, transform=train_transform)

    return data_dst

opts = get_argparser().parse_args()


def main():
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

    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    save_model_path = 'runs/' + opts.name
    utils.mkdir(save_model_path)
    # Restore

    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # load the pretrained models on voc
    for k in range(1, 10):
        csv_file = 'uav' + '{0:02}'.format(k)
        if(k <= 8):
            csv_file_next = 'uav' + '{0:02}'.format(k + 1)
        else:
            csv_file_next = None

        if(k == 1):
            label_dst = get_dataset_training(opts,  train_file='uav01_labeled.csv')
            label_loader = data.DataLoader(label_dst, batch_size=4, shuffle=True, num_workers=4)

        else:
            label_dst_1 = get_dataset_training(opts, train_file='uav01_labeled.csv')
            label_dst_2 = get_dataset_training(opts, train_file='pl_csv/airs.csv')

            concat_dataset = data.ConcatDataset([label_dst_1, label_dst_2])
            label_loader = data.DataLoader(
                concat_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=False)

        # pdb.set_trace()
        unlabel_dst = get_dataset_unlabel(opts, test_file=csv_file + '.csv')
        unlabel_loader = data.DataLoader(unlabel_dst, batch_size=4, shuffle=True, num_workers=4)

        labeled_train = iter(label_loader)
        unlabeled_train = iter(unlabel_loader)

        epoch_itrs = 100
        opts.total_itrs = opts.epochs * epoch_itrs
        opts.step_size = opts.total_itrs // 2
        # ################################################################################################load the pretrained models on voc

        model_voc = model_map[opts.model](
            num_classes=19, output_stride=opts.output_stride)
        checkpoint = torch.load(
            './runs/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar', map_location=torch.device('cpu'))
        model_voc.load_state_dict(checkpoint["model_state"])
        model = model_map[opts.model](
            num_classes=opts.num_classes, output_stride=opts.output_stride)

        pretrained_dict = model_voc.state_dict()
        model_dict = model.state_dict()
        del pretrained_dict['classifier.classifier.3.bias']
        del pretrained_dict['classifier.classifier.3.weight']
        pretrained_dict = {k: v for k,
                           v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        # if(k==1):
        #     model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
        #     checkpoint = torch.load('./runs/main_pl_semi_bl_4_4/0_best_deeplabv3plus_resnet101_airsim_os16.pth', map_location=torch.device('cpu'))
        #     model.load_state_dict(checkpoint["model_state"])

        if opts.separable_conv and 'plus' in opts.model:
            network.convert_to_separable_conv(model.classifier)
        utils.set_bn_momentum(model.backbone, momentum=0.01)
        metrics = StreamSegMetrics(opts.num_classes)

        optimizer = torch.optim.SGD(params=model.parameters(
        ), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

        #######################################################################

        if opts.lr_policy == 'poly':
            scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
        elif opts.lr_policy == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=opts.step_size, gamma=0.1)

        #######################################################################
        model = nn.DataParallel(model)
        model.to(device)
        model.train()

        best_score = 0.0

        for cur_epochs in range(opts.epochs):
            losses = utils.AverageMeter()
            losses_x = utils.AverageMeter()
            losses_u = utils.AverageMeter()
            ws = utils.AverageMeter()

            for cur_itrs in range(epoch_itrs):
                try:
                    inputs_x, targets_x, _, _ = labeled_train.next()
                except:
                    labeled_train = iter(label_loader)
                    inputs_x, targets_x, _, _ = labeled_train.next()

                try:
                    inputs_u, _ = unlabeled_train.next()

                except:
                    unlabeled_train = iter(unlabel_loader)
                    inputs_u, _ = unlabeled_train.next()

                batch_size = 4
                if(inputs_u.size(0) != batch_size or inputs_x.size(0) != batch_size):
                    break

                oh, ow = targets_x.size(1), targets_x.size(2)

                targets_x = targets_x.to(device).long()
                inputs_x = inputs_x.to(device)
                inputs_u = inputs_u.to(device)

                with torch.no_grad():
                    out_u = model(inputs_u)
                    out_u = F.interpolate(
                        out_u, size=[270, 480], mode='bilinear', align_corners=False)
                    targets_u = out_u.max(dim=1)[1].detach().long()

                idx = torch.randperm(inputs_u.size(0))
                input_a, input_b = inputs_x, inputs_u
                target_a, target_b = targets_x, targets_u

                for image_i in range(target_b.size(0)):
                    classes = torch.unique(target_b[image_i])
                    # classes = classes[classes != ignore_label]
                    nclasses = classes.shape[0]
                    classes = (classes[torch.Tensor(np.random.choice(nclasses, int(
                        (nclasses - nclasses % 2) / 2), replace=False)).long()]).cuda()
                    if image_i == 0:
                        MixMask = transformmasks.generate_class_mask(
                            target_b[image_i], classes).unsqueeze(0).cuda()
                    else:
                        MixMask = torch.cat((MixMask, transformmasks.generate_class_mask(
                            target_b[image_i], classes).unsqueeze(0).cuda()))

                MixMask = MixMask.view(target_a.size(
                    0), 1, target_a.size(1), target_a.size(2))

                mixed_input_u = MixMask * input_a + (1 - MixMask) * input_b
                mixed_target_u = MixMask.bool().sum(1) * target_a + \
                    (1 - MixMask.bool().sum(1)) * target_b

                outs_x = model(inputs_x)
                outs_u = model(mixed_input_u)
                outs_x = F.interpolate(
                    outs_x, size=[270, 480], mode='bilinear', align_corners=False)
                outs_u = F.interpolate(
                    outs_u, size=[270, 480], mode='bilinear', align_corners=False)

                Lx = criterion(outs_x, targets_x)
                Lu = criterion(outs_u, mixed_target_u)

                w = opts.lambda_u * \
                    linear_rampup(cur_epochs + cur_itrs / epoch_itrs)
                loss = Lx + w * Lu

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.update(loss.item(), inputs_x.size(0))
                losses_x.update(Lx.item(), inputs_x.size(0))
                losses_u.update(Lu.item(), inputs_x.size(0))
                ws.update(w, inputs_x.size(0))
                # if vis is not None:
                #     vis.vis_scalar('Loss', cur_itrs, np_loss)
                if (cur_itrs) % 100 == 0:
                    print("Epoch %d, Itrs %d/%d, Loss_x=%f, Loss_u=%f, ws=%f" %
                          (cur_epochs, cur_itrs, opts.total_itrs, losses_x.avg, losses_u.avg, ws.avg))

                scheduler.step()

        unlabel_dst = get_dataset_unlabel2(opts, test_file=csv_file + '.csv')
        unlabel_loader = data.DataLoader(unlabel_dst, batch_size=4, shuffle=True, num_workers=4)
        if(k == 1 or k == 9):
            predict(unlabel_loader, model, device, file='datasets/airs/pl_csv/airs.csv')
        else:
            predict(unlabel_loader, model, device, file=None)

        if(csv_file_next is not None):
            unlabel_dst2 = get_dataset_unlabel2(opts, test_file=csv_file_next + '.csv')
            unlabel_loader2 = data.DataLoader(unlabel_dst2, batch_size=4, shuffle=True, num_workers=4)
            predict(unlabel_loader2, model, device, file='datasets/airs/pl_csv/airs.csv')

        ## we use 1_airs as the ground only method for comparison
        save_ckpt(save_model_path + '/' + str(k) + '_airs.pth')


def predict(data_loader, model, device, file):
    model.eval()
    with torch.no_grad():
        for (images, img_name) in data_loader:
            images = images.to(device, dtype=torch.float32)

            outputs = model(images)
            outputs = F.interpolate(outputs, size=[270, 480], mode='bilinear', align_corners=False)
            # for selecting positive pseudo-labels
            out_prob = F.softmax(outputs)

            max_value, max_idx = torch.max(out_prob, dim=1)
            for i in range(len(images)):
                cv2.imwrite(img_name[i].replace('/images/', '/semantic_pl/'), max_idx[i, :, :].cpu().numpy())

                if file is not None:
                    with open(file, 'a') as f:
                        sample_new = img_name[i].replace(opts.data_root,'') + ',' + img_name[i].replace('/images/', '/semantic_pl/').replace(opts.data_root,'') + '\n'
                        f.write(sample_new)


def linear_rampup(current, rampup_length=opts.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)





if __name__ == '__main__':
    main()
