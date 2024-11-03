import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import KFold

def merge_files(output_file, part_files):

    with open(output_file, "wb") as merged_file:
        for part_file in part_files:
            with open(part_file, "rb") as file:
                merged_file.write(file.read())

# Example usage
part_files = ["iMach.pt.part1", "iMach.pt.part2", "iMach.pt.part3", "iMach.pt.part4","iMach.pt.part5", "iMach.pt.part6", "iMach.pt.part7", "iMach.pt.part8"]
merge_files("iMach.pt", part_files)

input = torch.load('./input.pt')
cd = torch.load('./cd.pt').reshape(-1, 1)
iMach = torch.load('./iMach.pt')

data = TensorDataset(input, iMach, cd)

kf = KFold(n_splits=10, shuffle=True, random_state=28)

import os
if not os.path.exists('data_fold/'):
    os.makedirs('data_fold/')

for i, (train_idx, test_idx) in enumerate(kf.split(data)):

    train_data = [data[j] for j in train_idx]
    test_data = [data[j] for j in test_idx]

    train_tensors = [torch.stack(tensors) for tensors in zip(*train_data)]
    test_tensors = [torch.stack(tensors) for tensors in zip(*test_data)]

    train_dataset = TensorDataset(*train_tensors)
    test_dataset = TensorDataset(*test_tensors)

    torch.save(train_dataset, f'./data_fold/train_data_fold_{i+1}.pt')
    torch.save(test_dataset, f'./data_fold/test_data_fold_{i+1}.pt')
    print(f"Fold {i+1} saved: Train samples - {len(train_idx)}, Test samples - {len(test_idx)}")

