import os
import argparse
from collections import Counter
from random import shuffle

import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from simclr_resnet import get_sresnet, name_to_params
import numpy as np
from sklearn.metrics import accuracy_score
from copy import deepcopy as dc
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
def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


import joblib

import torch

class LC(torch.nn.Module):
    def __init__(self, feature_dim, target_dim=40):
            super(LC, self).__init__()
            self.linear = torch.nn.Sequential(
                torch.nn.Linear(feature_dim, target_dim),
                torch.nn.Softmax()
            )

    def forward(self, input_dimension):
            return self.linear(input_dimension)


def construct_val(ilsvrc_path, return_sz=False):
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



    small_val = '/users/pyu12/scratch/ilsvrc_small_val/10k.jbl'

    data = joblib.load(small_val)

    unnoamlized_data = []
    for tup in data:
        unnoamlized_data.append(
            (inverse_normalize(tensor=torch.FloatTensor(tup[0]), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
             tup[1]))

    val_loader = torch.utils.data.DataLoader(
        unnoamlized_data,
        batch_size=2, shuffle=False, drop_last=False)

    if return_sz:
        return val_loader, len(unnoamlized_data)

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



def awa_pipeline():
    import joblib
    seen_tuples = joblib.load('/users/pyu12/data/pyu12/datasets/awa_bin/seen_np.jkl')

    # Join Data and Select 0.8 for train, 0.1 for test, val
    seen_inst = seen_tuples[0]
    seen_label = seen_tuples[2] - 1

    unnoamlized_inst = []
    for tup in seen_inst:
        unnoamlized_inst.append(
            inverse_normalize(tensor=torch.FloatTensor(tup), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))


    train_split = int(0.8 * len(seen_label))
    val_split = int(0.9 * len(seen_label))
    zipped = list(zip(unnoamlized_inst, seen_label))
    shuffle(zipped)
    train_tuples = zipped[:train_split]
    val_tuples = zipped[train_split:val_split]
    test_tuples = zipped[val_split:]

    train = DataLoader(train_tuples, shuffle=True, drop_last=False, batch_size=256)
    val = DataLoader(val_tuples, shuffle=True, drop_last=False, batch_size=64)
    test = DataLoader(test_tuples, shuffle=True, drop_last=False, batch_size=64)

    model = construct_simclr()


    #awa_eval('simclr', model, train, val, test)

save_loc = '/users/pyu12/scratch/ssl_exp/'



def extract_latent():
    name='simclr'
    device = 'cuda'
    data_loader, data_sz = construct_val('/users/pyu12/data/pyu12/datasets/ILSVRC', return_sz=True)
    # model, _ = get_resnet(*name_to_params(pth_path))
    # model.load_state_dict(torch.load(pth_path)['resnet'])
    model = construct_simclr()
    model = model.to(device).eval()
    latent_exp = np.zeros([data_sz, 100352])
    offset = 0
    for ndarray, label in data_loader:
        bz = len(label)
        imgs = torch.FloatTensor(ndarray).to(device)
        layers = model(imgs, output_layer=True)
        latent_exp[offset:offset+bz,:] = layers[3].detach().cpu().numpy()

    np.save(save_loc+name+'_final_latent_rep_100352', latent_exp)
    print('Completed')



def awa_eval(name, model, train, val, test, device='cuda:0'):
    # Load AwA Binary Data (~32-64GB)
    # Mix Seen and Unseen Select 10% test 10%eval
    # Fine-tune on animal category check test results
    loc = save_loc + 'awa'
    model = model.to(device).eval()

    lc_finetune_epoch = 120

    l1_dim = 802816
    l2_dim = 401408
    l3_dim = 200704
    l4_dim = 100352
    l5_dim = 2048

    layer1_model = LC(l1_dim, target_dim=40).to(device)
    layer2_model = LC(l2_dim, target_dim=40).to(device)
    layer3_model = LC(l3_dim, target_dim=40).to(device)
    layer4_model = LC(l4_dim, target_dim=40).to(device)
    layer5_model = LC(l5_dim, target_dim=40).to(device)

    opt1 = torch.optim.SGD(layer1_model.parameters(), lr=1e-4, momentum=0.9, nesterov=True)
    opt2 = torch.optim.SGD(layer2_model.parameters(), lr=1e-4, momentum=0.9, nesterov=True)
    opt3 = torch.optim.SGD(layer3_model.parameters(), lr=1e-4, momentum=0.9, nesterov=True)
    opt4 = torch.optim.SGD(layer4_model.parameters(), lr=1e-4, momentum=0.9, nesterov=True)
    opt5 = torch.optim.SGD(layer5_model.parameters(), lr=1e-4, momentum=0.9, nesterov=True)

    ce1 = torch.nn.CrossEntropyLoss()
    ce2 = torch.nn.CrossEntropyLoss()
    ce3 = torch.nn.CrossEntropyLoss()
    ce4 = torch.nn.CrossEntropyLoss()
    ce5 = torch.nn.CrossEntropyLoss()


    model.eval()
    a1b = -1
    a2b = -1
    a3b = -1
    a4b = -1
    a5b = -1

    epoch_val_acc = [[] for _ in range(5)]
    epoch_train_loss = [[] for _ in range(5)]

    for _ in tqdm(range(1, lc_finetune_epoch+1)):
        layer1_model.train()
        layer2_model.train()
        layer3_model.train()
        layer4_model.train()
        layer5_model.train()

        l1loss = []
        l2loss = []
        l3loss = []
        l4loss = []
        l5loss = []
        for ndarray, label in train:
            bz = len(label)
            imgs = torch.FloatTensor(ndarray).to(device)
            label = torch.LongTensor(label).to(device)
            layers = model(imgs, output_layer=True)
            # pass each batch into LC model

            opt1.zero_grad()
            l1pred = layer1_model(layers[0])
            loss1 = ce1(l1pred, label)
            l1loss.append(loss1.item())

            opt2.zero_grad()
            l2pred = layer2_model(layers[1])
            loss2 = ce2(l2pred, label)
            l2loss.append(loss2.item())

            opt3.zero_grad()
            l3pred = layer3_model(layers[2])
            loss3 = ce3(l3pred, label)
            l3loss.append(loss3.item())

            opt4.zero_grad()
            l4pred = layer4_model(layers[3])
            loss4 = ce4(l4pred, label)
            l4loss.append(loss4.item())

            opt5.zero_grad()
            l5pred = layer5_model(layers[4])
            loss5 = ce5(l5pred, label)
            l5loss.append(loss5.item())

            loss1.backward(retain_graph=True)
            loss2.backward(retain_graph=True)
            loss3.backward(retain_graph=True)
            loss4.backward(retain_graph=True)
            loss5.backward(retain_graph=True)

            opt4.step()
            opt3.step()
            opt2.step()
            opt1.step()
            opt5.step()

        epoch_train_loss[0].append(np.mean(l1loss))
        epoch_train_loss[1].append(np.mean(l2loss))
        epoch_train_loss[2].append(np.mean(l3loss))
        epoch_train_loss[3].append(np.mean(l4loss))
        epoch_train_loss[4].append(np.mean(l5loss))

        layer1_model.eval()
        layer2_model.eval()
        layer3_model.eval()
        layer4_model.eval()
        layer5_model.eval()

        l1_pred = []
        l2_pred = []
        l3_pred = []
        l4_pred = []
        l5_pred = []

        gt = []
        for ndarray, label in val:
            imgs = torch.FloatTensor(ndarray).to(device)
            layers = model(imgs, output_layer=True)
            # pass each batch into LC model
            l1pred = layer1_model(layers[0])
            l2pred = layer2_model(layers[1])
            l3pred = layer3_model(layers[2])
            l4pred = layer4_model(layers[3])
            l5pred = layer5_model(layers[4])
            _, l1top1 = torch.max(l1pred, 1)
            _, l2top1 = torch.max(l2pred, 1)
            _, l3top1 = torch.max(l3pred, 1)
            _, l4top1 = torch.max(l4pred, 1)
            _, l5top1 = torch.max(l5pred, 1)

            l1_pred += list(l1top1.detach().cpu().numpy())
            l2_pred += list(l2top1.detach().cpu().numpy())
            l3_pred += list(l3top1.detach().cpu().numpy())
            l4_pred += list(l4top1.detach().cpu().numpy())
            l5_pred += list(l4top1.detach().cpu().numpy())

            gt += list(label.detach().cpu().numpy())

        l1acc = accuracy_score(gt, l1_pred)
        l2acc = accuracy_score(gt, l2_pred)
        l3acc = accuracy_score(gt, l3_pred)
        l4acc = accuracy_score(gt, l4_pred)
        l5acc = accuracy_score(gt, l5_pred)

        epoch_val_acc[0].append(l1acc)
        epoch_val_acc[1].append(l2acc)
        epoch_val_acc[2].append(l3acc)
        epoch_val_acc[3].append(l4acc)
        epoch_val_acc[4].append(l5acc)

        if l1acc > a1b:
            l1_best_model = dc(layer1_model.state_dict())
            a1b = l1acc
        if l2acc > a2b:
            l2_best_model = dc(layer2_model.state_dict())
            a2b = l2acc
        if l3acc > a3b:
            l3_best_model = dc(layer3_model.state_dict())
            a3b = l3acc
        if l4acc > a4b:
            l4_best_model = dc(layer4_model.state_dict())
            a4b = l4acc
        if l5acc > a5b:
            l5_best_model = dc(layer5_model.state_dict())
            a5b = l5acc

    # Testing
    layer1_model.load_state_dict(l1_best_model)
    layer2_model.load_state_dict(l2_best_model)
    layer3_model.load_state_dict(l3_best_model)
    layer4_model.load_state_dict(l4_best_model)
    layer5_model.load_state_dict(l5_best_model)

    layer1_model.eval()
    layer2_model.eval()
    layer3_model.eval()
    layer4_model.eval()
    layer5_model.eval()

    l1_pred = []
    l2_pred = []
    l3_pred = []
    l4_pred = []
    l5_pred = []

    gt = []
    for ndarray, label in test:
        imgs = torch.FloatTensor(ndarray).to(device)
        layers = model(imgs, output_layer=True)
        # pass each batch into LC model
        l1pred = layer1_model(layers[0])
        l2pred = layer2_model(layers[1])
        l3pred = layer3_model(layers[2])
        l4pred = layer4_model(layers[3])
        l5pred = layer5_model(layers[4])

        _, l1top1 = torch.max(l1pred, 1)
        _, l2top1 = torch.max(l2pred, 1)
        _, l3top1 = torch.max(l3pred, 1)
        _, l4top1 = torch.max(l4pred, 1)
        _, l5top1 = torch.max(l5pred, 1)

        l1_pred += list(l1top1.detach().cpu().numpy())
        l2_pred += list(l2top1.detach().cpu().numpy())
        l3_pred += list(l3top1.detach().cpu().numpy())
        l4_pred += list(l4top1.detach().cpu().numpy())
        l5_pred += list(l5top1.detach().cpu().numpy())

        gt += list(label.detach().cpu().numpy())

    l1acc = accuracy_score(gt, l1_pred)
    l2acc = accuracy_score(gt, l2_pred)
    l3acc = accuracy_score(gt, l3_pred)
    l4acc = accuracy_score(gt, l4_pred)
    l5acc = accuracy_score(gt, l5_pred)

    test_acc = [l1acc, l2acc, l3acc, l4acc, l5acc]

    np.save(loc+name+'_epoch_losses', epoch_train_loss)
    np.save(loc+name+'_epoch_val_acc', epoch_val_acc)
    np.save(loc+name+'_test_acc', test_acc)
    print(name + ' Completed')


def run(pth_path, ig=True):

    device = 'cuda'
    data_loader = construct_val('/users/pyu12/data/pyu12/datasets/ILSVRC')
    # model, _ = get_resnet(*name_to_params(pth_path))
    # model.load_state_dict(torch.load(pth_path)['resnet'])
    model = construct_simclr()
    model = model.to(device).eval()
    preds = []
    target = []
    data_sz = 10000

    if ig:
        from captum.attr import IntegratedGradients, NoiseTunnel
        IG = IntegratedGradients(model)

    top_10_preds = np.zeros((data_sz, 10))
    if ig:
        ig_save = np.zeros((data_sz, 3, 224, 224))
    offset = 0
    for images, labels in tqdm(data_loader):
        bz = len(labels)
        imgs = images.to(device)
        res = model(imgs)
        _, pred = res.topk(1, dim=1)
        preds.append(pred.squeeze(1).cpu())
        target.append(labels)
        _, top10 = res.topk(10, dim=1)
        top_10_preds[offset:offset + bz, :] = top10.detach().cpu().numpy()
        if ig:
            ig_attributions = IG.attribute(imgs, target=pred.squeeze(1), n_steps=100)
            ig_save[offset:offset + bz, :, :, :] = ig_attributions.detach().cpu().numpy()

        offset += bz

    p = torch.cat(preds).numpy()
    t = torch.cat(target).numpy()
    # all_counters = [Counter() for i in range(1000)]
    # for i in range(50000):
    #     all_counters[t[i]][p[i]] += 1
    # total_correct = 0
    # for i in range(1000):
    #     total_correct += all_counters[i].most_common(1)[0][1]

    ground_truth = t
    predicted = p
    test_true_labels = np.array(ground_truth)
    test_results = np.array(predicted)

    eval_mode = 'micro'

    acc = accuracy_score(test_true_labels, test_results)

    print(acc)
    save_loc = '/users/pyu12/scratch/ssl_exp/'
    name = 'simclr'
    # np.save(save_loc + name + '_stats_10k', [acc, acc, acc, acc])
    # np.save(save_loc + name + '_ground_truth_10k', ground_truth)
    # np.save(save_loc + name + '_prediceted_10k', predicted)
    # np.save(save_loc + name + '_top10pred_10k', top_10_preds)
    if ig:
        np.save(save_loc+name+'_integrated_gradients_10k_redo', ig_save)


    total_correct = np.mean(p==t)
    print('ACC: {:.4f}'.format(total_correct))
    print(accuracy_score(p, t))



def process():
    name='simclr'
    original=np.load(save_loc + name + '_integrated_gradients_10k.npy')
    new = original[2:]
    np.save(save_loc + name + '_integrated_gradients_10k', new)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SimCLR verifier')
    parser.add_argument('pth_path', type=str, help='path of the input checkpoint file')
    args = parser.parse_args()
    #extract_latent()
    #run(args.pth_path, ig=False)
    process()
    #awa_pipeline()
