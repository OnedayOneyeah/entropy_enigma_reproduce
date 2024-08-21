import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import time

def plot_sub(cfg, 
             adapt = True):
    
    # open files
    if adapt:
        path = os.path.join(cfg.plot_dir, 'adapt_acc.pkl')
    else:
        path = os.path.join(cfg.plot_dir, 'test_acc.pkl')
        
    with open(path, 'rb') as f:
        acc_list = pkl.load(f)
        adapt_acc_arr = np.array(acc_list) * 100
        print(f"adapt acc list len: {len(adapt_acc_arr)}")
        
        
    # visualize
    xs = np.linspace(0, len(adapt_acc_arr), len(adapt_acc_arr))
    
    # save plot
    if adapt:
        m,b = np.polyfit(xs, adapt_acc_arr, 1)
        plt.plot(xs, adapt_acc_arr)
        plt.plot(xs, xs*m + b)
        plt.title('BATCH ACC (%) (by iters)')
    else:
        plt.plot(xs, acc_list)
        plt.title('TEST ACC (%) (by 10 iters)')
        
    plt.savefig("plot_{}_{}_{}.png".format(cfg.adaptation, cfg.top_k, time.strftime('%y%m%d_%X')), dpi = 300)
    
    

def plot(cfg):
    # plot_sub(cfg, adapt = True)
    plot_sub(cfg, adapt = False)