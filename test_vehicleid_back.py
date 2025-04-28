# -*- coding: utf-8 -*-
from __future__ import print_function, division

from cv2_transform import transforms
from torch.utils.data import DataLoader
import torch

from network.scm import SCM
from data_read import ImageTxtDataset

import argparse, time, os, sys, random
import numpy as np
from os import path as osp


def get_data(batch_size, test_set):
    transform_test = transforms.Compose([
        transforms.Resize(size=(opt.img_height, opt.img_width)),
        transforms.ToTensor(),
    ])

    test_imgs = ImageTxtDataset(test_set, transform=transform_test)

    test_data = DataLoader(test_imgs, batch_size, shuffle=False, num_workers=4)
    return test_data


def extract_feature(net, dataloaders):
    count = 0
    features = []
    for img, _ in dataloaders:
        n = img.shape[0]
        count += n
        print(count)
        ff = np.zeros((n, feat_num*(opt.num_part+3)), dtype=np.float32)
        for i in range(2):
            if(i==1):
                img = torch.flip(img, [3])
            with torch.no_grad():
                f = torch.cat(net(img.cuda()), dim=1).detach().cpu().numpy()
            ff = ff+f
        features.append(ff)
    features = np.concatenate(features)
    features = features / np.sqrt(np.sum(np.square(features), axis=1, keepdims=True))
    return features

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

    data_dir = osp.join(opt.dataset_root, "VehicleID_V1.0")
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus

    label_to_items = {}
    for line in open(osp.join(data_dir, "train_test_split", "test_list_800.txt")).readlines():
        name, label = line.strip().split()
        if label not in label_to_items.keys():
            label_to_items[label] = []
            
        label_to_items[label].append([
            osp.join(data_dir, 'image', name+".jpg"),
            label, 
            name
        ])
    
    test_set = []
    test_label = []
    test_cam = []

    for label in label_to_items.keys():
        items = label_to_items[label]
        random.shuffle(items)
        for i, item in enumerate(items):
            test_set.append(item[:2])
            test_label.append(item[1])
            test_cam.append(item[2])
    test_label = np.array(test_label)
    test_cam = np.array(test_cam)
   
    ######################################################################
    # Load Collected data Trained model
    mod_pth = osp.join('params', 'ema.pth')
    if opt.feat_num == 0:
        feat_num = 1024
    else:
        feat_num = opt.feat_num

    net = SCM(num_classes=13164, num_parts=[1,opt.num_part], feat_num=opt.feat_num, reduce_num=opt.reduce_num, net=opt.net) 
    
    net.load_state_dict(torch.load(mod_pth))
    net.cuda()
    net.eval()

    # Extract feature
    test_loader = get_data(opt.batch_size, test_set)
    print('start test')
    test_feature = extract_feature(net, test_loader)

    num = test_label.size
    dist_all = np.dot(test_feature, test_feature.T)

    CMC = np.zeros(test_label.size - 1)
    ap = 0.0
    for i in range(num):
        cam = test_cam[i]
        label = test_label[i]
        dist_tmp = np.concatenate((dist_all[i][:i], dist_all[i][i+1:]), axis=0)
        index = np.argsort(-dist_tmp)

        test_label_tmp = np.concatenate((test_label[:i], test_label[i+1:]), axis=0)
        query_index = np.argwhere(test_label_tmp==label)
        test_cam_tmp = np.concatenate((test_cam[:i], test_cam[i+1:]), axis=0)
        camera_index = np.argwhere(test_cam_tmp==cam)

        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        junk_index = np.intersect1d(query_index, camera_index)
    
        ap_tmp, CMC_tmp = compute_mAP(index, good_index, junk_index)
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC/num #average CMC
    print('top1:%f top5:%f top10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/num))
