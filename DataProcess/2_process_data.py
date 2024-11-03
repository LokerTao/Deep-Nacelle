import torch
from torch.utils.data import TensorDataset
import numpy as np
import os

def process_data(path):

    train_dataset = torch.load('./data_fold/train_data_fold_%s.pt'%(str(path)))
    test_dataset = torch.load('./data_fold/test_data_fold_%s.pt'%(str(path)))
    
    
    train_input = train_dataset.tensors[0]
    train_iMach = train_dataset.tensors[1]
    train_cd = train_dataset.tensors[2]
    
    
    test_input = test_dataset.tensors[0]
    test_iMach = test_dataset.tensors[1]
    test_cd = test_dataset.tensors[2]
    
    '''
    scale input
    '''

    min_vals = np.array([0.6, 1.1, 0.28, 9, 0.6, 1.1, 0.28, 9, 0.6, 1.1, 0.28, 9])
    max_vals = np.array([1.5, 1.3, 0.45, 15, 1.5, 1.3, 0.45, 15, 1.5, 1.3, 0.45, 15])

    train_input = (train_input - min_vals) / (max_vals - min_vals)

    test_input = (test_input - min_vals) / (max_vals - min_vals)

    '''
    scale cd
    '''

    min_vals = np.array([0])
    max_vals = np.array([0.205])

    train_cd = (train_cd - min_vals) / (max_vals - min_vals)
    
    test_cd = (test_cd - min_vals) / (max_vals - min_vals)

    '''
    scale iMach
    '''
    
    iMach_mean = train_iMach.mean([0, 1, 2])
    iMach_std = train_iMach.std([0, 1, 2])

    iMach_train = (train_iMach - iMach_mean) / iMach_std
    iMach_test = (test_iMach - iMach_mean) / iMach_std

    '''
    save data
    '''

    train_input = train_input.to(dtype=torch.float32)
    test_input = test_input.to(dtype=torch.float32)

    train_cd = train_cd.to(dtype=torch.float32)
    test_cd = test_cd.to(dtype=torch.float32)

    train_dataset = TensorDataset(train_input, iMach_train, train_cd)
    test_dataset = TensorDataset(test_input, iMach_test, test_cd)
    
    if not os.path.exists("data_fold/%s"%(str(path))):

        os.makedirs("data_fold/%s"%(str(path)))

    torch.save(train_dataset, 'data_fold/%s/train_dataset.pth'%(str(path)))
    torch.save(test_dataset, 'data_fold/%s/test_dataset.pth'%(str(path)))

    np.save('data_fold/%s/input_train_scaled.npy'%(str(path)), train_input.numpy())
    np.save('data_fold/%s/input_test_scaled.npy'%(str(path)), test_input.numpy())
    np.save('data_fold/%s/cd_train_scaled.npy'%(str(path)), train_cd.numpy())
    np.save('data_fold/%s/cd_test_scaled.npy'%(str(path)), test_cd.numpy())
    np.save('data_fold/%s/iMach_train_scaled.npy'%(str(path)), iMach_train.numpy())
    np.save('data_fold/%s/iMach_test_scaled.npy'%(str(path)), iMach_test.numpy())
    np.save('data_fold/%s/iMach_mean.npy'%(str(path)),iMach_mean.numpy())
    np.save('data_fold/%s/iMach_std.npy'%(str(path)),iMach_std.numpy())

if __name__ == "__main__":

    for j in range(1,11):

        process_data(j)
