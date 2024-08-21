import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import time

def plot_sub(cfg, 
             adapt = False):
    
    # open files
    if adapt:
        path = os.path.join(cfg.plot_dir, 'adapt_acc.pkl')
    else:
        path = os.path.join(cfg.plot_dir, 'test_acc.pkl')
        
    with open(path, 'rb') as f:
        acc_list = pkl.load(f)
        adapt_acc_list = np.array(acc_list)
        print(f"adapt acc list len: {len(acc_list)}")
        
        
    # visualize
    xs = np.linspace(0, len(acc_list), len(acc_list))
    
    # save plot
    plt.plot(xs, acc_list)
    plt.savefig("plot_{}_{}_{}.png".format(cfg.adaptation, cfg.top_k, time.strftime('%y%m%d_%X')), dpi = 300)
    
    

def plot(cfg):
    # plot_sub(cfg, adapt = True)
    plot_sub(cfg, adapt = False)