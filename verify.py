import os
import argparse
from collections import Counter

import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from simclr_resnet import get_sresnet, name_to_params
import numpy as np
from sklearn.metrics import accuracy_score
# class ImagenetValidationDataset(Dataset):
#     def __init__(self, val_path):
#         super().__init__()
#         self.val_path = val_path
#         self.transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
#         with open(os.path.join(val_path, 'ILSVRC2012_validation_ground_truth.txt')) as f:
#             self.labels = [int(l) - 1 for l in f.readlines()]
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, item):
#         img = Image.open(os.path.join(self.val_path, f'ILSVRC2012_val_{item + 1:08d}.JPEG')).convert('RGB')
#         return self.transform(img), self.labels[item]

def construct_val(ilsvrc_path):
    # valdir = os.path.join(ilsvrc_path, 'val')
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    #
    # val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor()
    #     #normalize,
    # ]))
    #
    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=256, shuffle=False,
    #     num_workers=4, pin_memory=True)

    def inverse_normalize(tensor, mean, std):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor
    import joblib

    small_val = '/users/pyu12/scratch/ilsvrc_small_val/10k.jbl'

    data = joblib.load(small_val)

    unnoamlized_data = []
    for tup in data:
        unnoamlized_data.append(
            (inverse_normalize(tensor=torch.FloatTensor(tup[0]), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
             tup[1]))

    val_loader = torch.utils.data.DataLoader(
        unnoamlized_data,
        batch_size=256, shuffle=False, pin_memory=True)

    return val_loader


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t().cpu()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum().item()
        res.append(correct_k)
    return res

ccv_simclr_path = '/users/pyu12/scratch/ilsvrc_small_val/adjusted_simclr_r50.pth'
def construct_simclr(pth_path=ccv_simclr_path):
    model, _ = get_sresnet(50, 1, 0)
    model.load_state_dict(torch.load(pth_path))
    return model


@torch.no_grad()
def run(pth_path):
    device = 'cuda'
    data_loader = construct_val('/users/pyu12/data/pyu12/datasets/ILSVRC')
    # model, _ = get_resnet(*name_to_params(pth_path))
    # model.load_state_dict(torch.load(pth_path)['resnet'])
    model = construct_simclr()
    model = model.to(device).eval()
    preds = []
    target = []
    for images, labels in tqdm(data_loader):
        _, pred = model(images.to(device)).topk(1, dim=1)
        preds.append(pred.squeeze(1).cpu())
        target.append(labels)
    p = torch.cat(preds).numpy()
    t = torch.cat(target).numpy()
    # all_counters = [Counter() for i in range(1000)]
    # for i in range(50000):
    #     all_counters[t[i]][p[i]] += 1
    # total_correct = 0
    # for i in range(1000):
    #     total_correct += all_counters[i].most_common(1)[0][1]

    total_correct = np.mean(p==t)
    print('ACC: {:.4f}'.format(total_correct))
    print(accuracy_score(p, t))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SimCLR verifier')
    parser.add_argument('pth_path', type=str, help='path of the input checkpoint file')
    args = parser.parse_args()
    run(args.pth_path)
