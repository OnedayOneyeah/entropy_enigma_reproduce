# import modules
import os
from tqdm import tqdm
import pickle as pkl
import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

def make_data(data_dir:str = '/home/s2/ywhan/data',
              index_dir: str = './data',
              holdouts_ratio: int = 0.3,
              top_k:int = 1,
              epochs:int = 7,
              test = False
              ):       
    
    # make holdouts sets
    batch_size = 512
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data = datasets.ImageFolder(data_dir, 
                                      transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]))
    num_data = len(data)
    indices = list(range(num_data))
    np.random.shuffle(indices)
    split = int(np.floor(holdouts_ratio*num_data))
    val_idx, test_idx = indices[split:], indices[:split]
    
    # save the splitted indices
    val_path = os.path.join(index_dir, 'val_idx.pkl')
    test_path = os.path.join(index_dir, 'test_idx.pkl')
    with open(val_path, 'wb') as f:
        pkl.dump(val_idx, f)
    with open(test_path, 'wb') as f:
        pkl.dump(test_idx, f)
    
    
    data_loader = torch.utils.data.DataLoader(data,
                                            batch_size=batch_size, 
                                            num_workers=4, pin_memory=True, drop_last=False,
                                            shuffle = False)
    
    # select indices w. pretrained resnet50
    if top_k > 1 :
        raise NotImplementedError
    else:
        selected_idx = [] 
    
    batch_size = 512
        
    # load model
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Device: %s" % DEVICE)
    
    model = models.resnet50(weights="IMAGENET1K_V2") # New weights with accuracy 80.858%
    model = model.to(DEVICE)
    model.eval() 
    with torch.no_grad():
        for i in range(epochs):
            tqdm_bar = tqdm(data_loader)
            correct = 0
            n_class = 1000
            class_wise_count = [0. for i in range(n_class)] 
            class_wise_corr = [0. for i in range(n_class)]
            
            print(f"===== EPOCHS: {i} =====")
            for i, (images, labels) in enumerate(tqdm_bar):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                # inference
                preds = model(images)
                predictions = preds.max(1)[1]
                            
                # select index
                dummies = torch.arange(end = predictions.shape[0]).to(DEVICE)
                wrong_idx = [idx.item() + batch_size * i for idx in predictions.ne(labels.view_as(predictions)) * dummies if idx != 0]
                selected_idx.extend(wrong_idx)
                
                # evaluate
                correct += predictions.eq(labels.view_as(predictions)).sum().item()
                
                for pred, label in zip(predictions, labels):
                    class_wise_count[label.item()] += 1
                    if pred.item() == label.item():
                        class_wise_corr[label.item()] += 1
            
                if i%10 == 0:
                    print(selected_idx)
            
            acc = 100. * correct / len(data_loader.dataset)
            print(f'PRETRAINED RESENT50 ACC on IMAGENET-C VALSET : {acc:.2f}(%)') # 46.87%
            class_wise_corr, class_wise_count = np.array(class_wise_corr), np.array(class_wise_count)
            class_wise_acc = class_wise_corr/class_wise_count
            # print(f'PRETRAINED RESENT50 ACC on IMAGENET-C VALSET : {acc:.2f}')
            df = pd.DataFrame({'class_wise_acc': class_wise_acc.round(2)})
            print(f'CLASS_WISE_ACC : \n{df}')
            
    
    if not test:
        path = os.path.join(index_dir, f'top_{top_k}_indices.pkl')
        with open(path, 'wb') as f:
            pkl.dump(selected_idx, f)
            print(f"Selected indices (top_{top_k}) are saved.")


def load_data_top_k(data_dir:str, 
              index_dir:str,
              top_k:int,
              batch_size:int):
    
    # load val data
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data = datasets.ImageFolder(data_dir, 
                                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]))
    # load val split indices
    path = os.path.join(index_dir, 'val_idx.pkl')
    with open(path, 'rb') as f:
        val_idx = pkl.load(f)
    
    # load selected indices
    path = os.path.join(index_dir, f'top_{top_k}_indices.pkl')
    with open(path, 'rb') as f:
        selected_idx = pkl.load(f)
    
    adapt_idx = [idx for idx in val_idx if idx in selected_idx]
            
    adapt_data = torch.utils.data.Subset(data, adapt_idx)
    print(f"# of {top_k} samples: {len(adapt_data)}") # 18532
    assert adapt_data.indices == adapt_idx, 'Selected incides are not aligned'
    adapt_data_loader = torch.utils.data.DataLoader(adapt_data,
                                                batch_size=batch_size, 
                                            num_workers=4, pin_memory=True, drop_last=False)
    
    return adapt_data_loader


def load_data_test(data_dir:str = '/home/s2/ywhan/data',
                   index_dir:str = './data',
                   top_k:int = 1,
                   batch_size:int = 512,
                   ):
    
    # load val data
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    data = datasets.ImageFolder(data_dir, 
                                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]))
    # load val split indices
    path = os.path.join(index_dir, 'test_idx.pkl')
    with open(path, 'rb') as f:
        test_idx = pkl.load(f)
        
    test_data = torch.utils.data.Subset(data, test_idx)
    test_data_loader = torch.utils.data.DataLoader(test_data,
                                                batch_size=batch_size, 
                                            num_workers=4, pin_memory=True, drop_last=False)
    
    return test_data_loader