# -*- coding: utf-8 -*-
from __future__ import print_function, division

from cv2_transform import transforms
from torch.utils.data import DataLoader
import torch

from network.scm import SCM
from data_read import ImageTxtDataset

import os, argparse
from os import path as osp
from collections import defaultdict
import numpy as np
import scipy.io as sio
from sklearn.metrics import normalized_mutual_info_score
import faiss

def get_data(batch_size, test_set):
    transform_test = transforms.Compose([
        transforms.Resize(size=(opt.img_height+32, opt.img_width+32)),
        transforms.CenterCrop(size=(opt.img_height, opt.img_width)),
        transforms.ToTensor(),
    ])

    test_imgs = ImageTxtDataset(test_set, transform=transform_test)
    test_data = DataLoader(test_imgs, batch_size, shuffle=False, num_workers=8)
    return test_data

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

def get_cluster_labels(x, nmb_clusters):
    dim = x.shape[1]

    # faiss implementation of k-means
    clus = faiss.Clustering(dim, nmb_clusters)
    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    index = faiss.IndexFlatL2(dim)
    # if faiss.get_num_gpus() > 0:
    #     index = faiss.index_cpu_to_all_gpus(index)
    # perform the training
    clus.train(x, index)
    _, idxs = index.search(x, 1)

    return [int(n[0]) for n in idxs]

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

    data_dir = osp.join(opt.dataset_root, "CUB_200_2011")
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus

    label_to_items = defaultdict(list)
    
    names_lines = open(osp.join(data_dir, "images.txt")).readlines()
    labels_lines = open(osp.join(data_dir, "image_class_labels.txt")).readlines()
    lines = [[name.strip().split()[1], int(label.strip().split()[1])]for (name, label) in zip(names_lines, labels_lines)]

    test_set = []
    test_label = []

    for idx in range(len(lines)):
        name, label = lines[idx]
        if label > 100:
            test_set.append([osp.join(data_dir, "images", name), label])
            test_label.append(label)

    print(len(test_set))
    test_label = np.array(test_label)

    ######################################################################
    # Load Collected data Trained model
    mod_pth = osp.join('params', 'ema.pth')
    if opt.feat_num == 0:
        opt.feat_num = 1024

    net = SCM(num_classes=100, num_parts=[1,opt.num_part], feat_num=opt.feat_num, reduce_num=opt.reduce_num, net=opt.net, h=opt.img_height//16, w=opt.img_width//16)

    net.load_state_dict(torch.load(mod_pth), strict=False)
    net.cuda()
    net.eval()

    # Extract feature
    test_loader = get_data(opt.batch_size, test_set)
    test_feature = extract_feature(net, test_loader)

    cluster_labels = get_cluster_labels(test_feature, 100)
    nmi = normalized_mutual_info_score(test_label, cluster_labels)

    num = test_label.size
    dist_all = np.dot(test_feature, test_feature.T)

    K = [1,2,4,8]
    recall_k = np.zeros(num)
    for i in range(num):
        label = test_label[i]
        pt_dist = dist_all[i]
        pt_index = np.argsort(-pt_dist)[1:max(K)+1]
        pt_label = test_label[pt_index]
        for k in K:
            recall_k[k-1] += ((pt_label[:k] == label).astype(np.float32).sum() >= 1).astype(np.float32)
    recall_k = recall_k / num

    print('Recall@1:%f Recall@2:%f Recall@4:%f Recall@8:%f NMI:%f'%(recall_k[0], recall_k[1], recall_k[3], recall_k[7], nmi))
