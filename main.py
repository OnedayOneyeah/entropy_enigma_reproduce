"""
GOAL: ENTROPY ENIGMA REPRODUCE (https://arxiv.org/abs/2405.05012)
WARNING: check if the pretrained model was train with data augmentation with gaussian noise corruptions

==CONFIG==========================================
- arch: resnet50
- pretrained ckpt: imagenet1k (v2, provided by torch with accuracy 80.858%)
- topk: 1 (by default)
- holdouts ratio: 0.3 (by default)
- data: IMAGENET-C (1k) - gaussian noise (severity 3)
- evaluation: every 10 iter (bs: 128 - TENT default setting)
"""

# import modules
import argparse
import os
from utils import set_seed
from data import make_data, load_data_top_k, load_data_test
from model import build_model
from visualize import plot
import pickle as pkl
import torch
import torchvision.datasets as datasets
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import logging

logger = logging.getLogger(__name__)

def adapt(adapt_loader,
          test_loader,
          model,
          cfg):
    
    device = cfg.device
    batch_size = cfg.batch_size
     
    acc = 0.
    total_cnt = 0
    adapt_acc_list = []
    test_acc_list = []
    
    for i, (images, labels) in enumerate(adapt_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images, if_adapt = True) # adaptation
        # Metrics
        _, preds = outputs.max(1)
        
        batch_acc = (preds == labels).float().sum()
        batch_cnt = labels.shape[0]
        adapt_acc_list.append(batch_acc.item()/batch_cnt)
        
        # entire acc, batch
        acc += batch_acc
        total_cnt += batch_cnt
        
        if i%10 == 0:
            test_acc = evaluate(test_loader, model, cfg)
            test_acc_list.append(test_acc)
            print(f'TEST ACC at {i}-th iters: {test_acc*100:.2f} (%)')
    
    # save accs
    if not os.path.isdir(cfg.plot_dir):
        os.mkdir(cfg.plot_dirs)
        
    adapt_acc_path = os.path.join(cfg.plot_dir, 'adapt_acc.pkl')
    test_acc_path = os.path.join(cfg.plot_dir, 'test_acc.pkl')
    
    with open(adapt_acc_path, 'wb') as f:
        pkl.dump(adapt_acc_list, f)
    
    with open(test_acc_path, 'wb') as f:
        pkl.dump(test_acc_list, f)
        
    return acc.item() / total_cnt
    

def evaluate(test_loader,
             model,
             cfg,
             ):
    
    acc = 0.
    total_cnt = 0
    device = cfg.device
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, if_adapt = False)  
            _, preds = outputs.max(1)
            
            # entire acc, batch
            acc += (preds == labels).float().sum()
            total_cnt += labels.shape[0]
            
        return acc.item()/total_cnt
    

def main(cfg):
    # setting
    set_seed(cfg)
    
    # make data split, and select indices for wronly classified samples
    if not os.path.isfile(os.path.join(cfg.index_dir, f'top_{cfg.top_k}_indices.pkl')):
        make_data(cfg.data_dir, cfg.index_dir, 
                cfg.holdouts_ratio,
                cfg.top_k, cfg.n_epochs)

    # load data loader
    adapt_loader = load_data_top_k(cfg.data_dir, cfg.index_dir, cfg.top_k, cfg.batch_size)
    test_loader = load_data_test(cfg.data_dir, cfg.index_dir, cfg.top_k, cfg.batch_size)

    # load model
    # load model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Device: %s" % device)
    base_model = models.resnet50(weights="IMAGENET1K_V2").to(device)
    cfg.device = device
    model = build_model(base_model, cfg, logger)
    print("TENT successfully loaded")
    
    # # check params
    # for n, m in model.named_modules():
    #     for np, p in m.named_parameters():
    #         # print(n)
    #         if p.requires_grad == True:
    #             # print(isinstance(m, nn.BatchNorm2d))
    #             print(np)
                   
    # exit()
    
    # adaptation
    adapt(adapt_loader, test_loader, model, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        default = '/shared/s2/lab01/dataset/data/ImageNet-C/gaussian_noise/3',
                        help = 'imagenet-c directory')
    
    # selecting indices
    parser.add_argument('--holdouts_ratio',
                        default = 0.3,
                        help = 'ratio for holdout sets')
    parser.add_argument('--top_k',
                        default = 1,
                        help = 'top k samples are excluded.')
    parser.add_argument('--n_epochs',
                        default = 1,
                        help = 'epochs to select indicies')
    parser.add_argument('--index_dir',
                        default = '/home/s2/ywhan/ent/data',
                        help = 'subset idxs directory')
    
    # adaptation
    parser.add_argument('--adaptation',
                        default = 'TENT',
                        help = 'TTA method')
    parser.add_argument('--batch_size',
                        default = 128, # TENT setting
                        help = 'batch size for TTA')
    # utils
    parser.add_argument('--seed',
                        default = 1,
                        help = 'random seed')
    
    # optimizer options
    parser.add_argument('--optimizer',
                        default = 'adam',
                        help = 'optimizer type')
    parser.add_argument('--lr',
                        default = 1e-3, # TENT setting
                        help = 'learning rate')
    parser.add_argument('--beta',
                        default = 0.9,
                        help = 'optimizer: beta for ADAM')
    parser.add_argument('--wd',
                        default = 0.0,
                        help = 'optimizer: l2 regularization')
    parser.add_argument('--momentum',
                        default = 0.9,
                        help = 'optimizer: momentum for SGD')
    parser.add_argument('--nestrov',
                        default = True,
                        help = 'optimizer: Nesterov momentum for SGD')
    
    # visualize
    parser.add_argument('--plot',
                        action = 'store_true',                        
                        help = 'visualize the accuracy plot')
    parser.add_argument('--plot_dir',
                        default = './plots',
                        help = 'plot directory')
    cfg = parser.parse_args()
    # Benchmark to select fastest CUDNN algorithms (best for fixed input sizes)
    cfg.CUDNN_BENCHMARK = True
    cfg.SAVE_DIR = "./output"
    cfg.LOG_DEST = "log.txt"


    print(cfg)
    main(cfg)
    
    if cfg.plot:
        plot(cfg)
        
    
    
    
    