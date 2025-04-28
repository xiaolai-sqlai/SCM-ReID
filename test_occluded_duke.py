# -*- coding: utf-8 -*-
from __future__ import print_function, division

from cv2_transform import transforms
from torch.utils.data import DataLoader
import torch

from network.scm import SCM
from data_read import ImageTxtDataset

import argparse, time, os, sys
import numpy as np
from os import path as osp
from matplotlib import cm
import matplotlib.pyplot as plt

def get_data(batch_size, test_set, query_set):
    transform_test = transforms.Compose([
        transforms.Resize(size=(opt.img_height, opt.img_width)),
        transforms.ToTensor(),
    ])

    test_imgs = ImageTxtDataset(test_set, transform=transform_test)
    query_imgs = ImageTxtDataset(query_set, transform=transform_test)

    test_data = DataLoader(test_imgs, batch_size, shuffle=False, num_workers=4)
    query_data = DataLoader(query_imgs, batch_size, shuffle=False, num_workers=4)
    return test_data, query_data


def extract_mask(net, dataloaders):
    count = 0
    output_dir = 'output_occluded_duke'
    os.makedirs(output_dir, exist_ok=True)

    for img, _ in dataloaders:
        competitive_masks = None
        cooperative_mask = None

        with torch.no_grad():
            _, competitive_masks, cooperative_mask = net(img.cuda())

        B, _, H, W = img.shape

        for i in range(B):
            img_subdir = os.path.join(output_dir, f'image_{count}')
            os.makedirs(img_subdir, exist_ok=True)

            # Save competitive masks
            for idx, mask in enumerate(competitive_masks):
                mask_resized = torch.nn.functional.interpolate(mask[i:i+1], size=(H, W), mode='bicubic', align_corners=False)
                mask_np = mask_resized.squeeze().cpu().numpy()

                cmap = cm.get_cmap('jet')
                mask_colored = (255 * cmap(mask_np)[:, :, 1:]).clip(0, 255).astype(np.uint8)

                original_img = (255 * img[i]).permute(1,2,0).cpu().numpy().clip(0, 255).astype(np.uint8)
                plt.imsave(os.path.join(img_subdir, f'competitive_{idx}.png'), (0.6 * original_img + 0.4 * mask_colored).clip(0, 255).astype(np.uint8))

            # save cooperative mask
            mask_resized = torch.nn.functional.interpolate(cooperative_mask[i:i+1], size=(H, W), mode='bicubic', align_corners=False)
            mask_np = mask_resized.squeeze().cpu().numpy()

            cmap = cm.get_cmap('jet')
            mask_colored = (255 * cmap(mask_np)[:, :, 1:]).clip(0, 255).astype(np.uint8)

            original_img = (255 * img[i]).permute(1,2,0).cpu().numpy().clip(0, 255).astype(np.uint8)
            plt.imsave(os.path.join(img_subdir, f'cooperative.png'), (0.6 * original_img + 0.4 * mask_colored).clip(0, 255).astype(np.uint8))

            count += 1
        break
    return None


def overlay_mask(img, mask, colormap='jet', alpha=0.4):
    if not isinstance(img, PIL.Image.Image) or not isinstance(mask, PIL.Image.Image):
        raise TypeError('img and mask arguments need to be PIL.Image')

    if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
        raise ValueError('alpha argument is expected to be of type float between 0 and 1')

    cmap = cm.get_cmap(colormap)
    # Resize mask and apply colormap
    overlay = mask.resize(img.size, resample=PIL.Image.BICUBIC)
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, 1:]).astype(np.uint8)
    # Overlay the image with the mask
    overlayed_img = PIL.Image.fromarray((alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8))

    return overlayed_img


def extract_feature(net, dataloaders):
    count = 0
    features = []
    for img, _ in dataloaders:
        n = img.shape[0]
        count += n
        print(count)
        ff = np.zeros((n, opt.feat_num*(opt.num_part+3)), dtype=np.float32)
        for i in range(2):
            if(i==1):
                img = torch.flip(img, [3])
            with torch.no_grad():
                f = torch.cat(net(img.cuda())[0], dim=1).detach().cpu().numpy()
            ff = ff+f
        features.append(ff)
    features = np.concatenate(features)
    features = features / np.sqrt(np.sum(np.square(features), axis=1, keepdims=True))
    return features


def get_id(img_path):
    cameras = []
    labels = []
    for path in img_path:
        cameras.append(int(path[0].split('/')[-1].split('_')[1][1]))
        labels.append(path[1])
    return np.array(cameras), np.array(labels)


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = np.zeros(len(index))
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--img-height', type=int, default=384)
    parser.add_argument('--img-width', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--dataset-root', type=str, default="../dataset/")
    parser.add_argument('--net', type=str, default="regnet_y_3_2gf", help="regnet_y_3_2gf, regnet_y_8gf")
    parser.add_argument('--gpus', type=str, default="0,1", help='number of gpus to use.')
    parser.add_argument('--num-part', type=int, default=2)
    parser.add_argument('--feat-num', type=int, default=0)
    parser.add_argument('--reduce-num', type=int, default=64)

    opt = parser.parse_args()

    data_dir = osp.join(opt.dataset_root, "Occluded_Duke")
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus


    test_set = [(osp.join(data_dir, 'bounding_box_test',line), int(line.split('_')[0])) for line in os.listdir(osp.join(data_dir, 'bounding_box_test')) if "jpg" in line and "-1" not in line]
    query_set = [(osp.join(data_dir, 'query',line), int(line.split('_')[0])) for line in os.listdir(osp.join(data_dir, 'query')) if "jpg" in line]
    
    test_cam, test_label = get_id(test_set)
    query_cam, query_label = get_id(query_set)

    ######################################################################
    # Load Collected data Trained model
    mod_pth = osp.join('params', 'ema.pth')
    if opt.feat_num == 0:
        opt.feat_num = 1024

    net = SCM(num_classes=702, num_parts=[1,opt.num_part], feat_num=opt.feat_num, reduce_num=opt.reduce_num, net=opt.net, h=opt.img_height//16, w=opt.img_width//16)
 
    net.load_state_dict(torch.load(mod_pth))
    net.cuda()
    net.eval()

    # Extract feature
    test_loader, query_loader = get_data(opt.batch_size, test_set, query_set)
    print('start test')
    test_feature = extract_feature(net, test_loader)
    print('start query')
    query_feature = extract_feature(net, query_loader)

    # Extract masks
    extract_mask(net, test_loader)

    num = query_label.size
    dist_all = np.dot(query_feature, test_feature.T)

    CMC = np.zeros(test_label.size)
    ap = 0.0
    for i in range(num):
        cam = query_cam[i]
        label = query_label[i]
        index = np.argsort(-dist_all[i])

        query_index = np.argwhere(test_label==label)
        camera_index = np.argwhere(test_cam==cam)

        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        junk_index = np.intersect1d(query_index, camera_index)
    
        ap_tmp, CMC_tmp = compute_mAP(index, good_index, junk_index)
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC/num #average CMC
    print('top1:%f top5:%f top10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/num))
