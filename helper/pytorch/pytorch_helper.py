import os
import pickle
from collections import OrderedDict
import numpy as np
import torch
from PIL import Image
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import VisionDataset
from typing import Any, Callable, Optional, Tuple
from helper.pytorch.dataset import ImageNetDataset


class PytorchHelper:

    def increment_average(self, model, model_next, n):
        """ Update an incremental average. """
        if type(model) == list:
            w = []
            for i in range(len(model)):
                temp = OrderedDict()
                for name in model[i].keys():
                    tensor_diff = model_next[i][name] - model[i][name]
                    temp[name] = model[i][name] + tensor_diff / n
                w.append(temp)
            return w
        w = OrderedDict()
        for name in model.keys():
            tensor_diff = model_next[name] - model[name]
            w[name] = model[name] + tensor_diff / n
        return w

    def get_tensor_diff(self, model, base_model):
        w = OrderedDict()
        for name in model:
            w[name] = model[name] - base_model[name]
        return w

    def add_base_model(self, tensordiff, base_model, learning_rate):
        w = OrderedDict()
        for name in tensordiff:
            w[name] = learning_rate * tensordiff[name] + base_model[name]
        return w

    def save_model(self, weights_dict, path=None):
        # import pickle
        # with open(path, "wb") as tf:
        #     pickle.dump(weights_dict, tf)
        if not path:
            path = self.get_tmp_path()
        np.savez_compressed(path, **weights_dict)
        return path

    def load_model(self, path=None):
        # import pickle
        # with open("initial_model.pkl", "rb") as tf:
        #     b = pickle.load(tf)
        #     print(type(b))
        b = np.load(path, allow_pickle=True)
        # print(list(b.keys()))
        # if "momentum_change" in b.keys():
        #     return b["model"], b["momentum"], b["momentum_change"]
        # return b["model"], b["momentum"]
        weights_np = OrderedDict()
        for i in b.files:
            weights_np[i] = b[i]
        momentum = weights_np["momentum"]
        weights_np.pop("momentum")
        if "momentum_change" in weights_np.keys():
            momentum_change = weights_np["momentum_change"]
            weights_np.pop("momentum_change")
            return weights_np, momentum, momentum_change
        return weights_np, momentum

    def read_data(self, dataset, data_path, samples, trainset):
        if dataset == "imagenet":
            return self.read_data_imagenet(data_path, trainset, samples)
        elif dataset == "mnist":
            return self.read_data_mnist(trainset, samples)
        elif dataset == "cifar10":
            return self.read_data_cifar10(trainset, samples)
        elif dataset == "cifar100":
            return self.read_data_cifar100(trainset, samples)

    def read_data_cifar100(self, trainset=True, samples=5000):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if trainset:
            dataset = datasets.CIFAR100(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True)
            sample = np.random.permutation(len(dataset))[:samples]
        else:
            dataset = datasets.CIFAR100(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
            sample = np.random.permutation(len(dataset))[:samples]
        return torch.utils.data.Subset(dataset, sample)

    def read_data_cifar10(self, trainset=True, samples=5000):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if trainset:
            dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True)
            sample = np.random.permutation(len(dataset))[:samples]
        else:
            dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
            sample = np.random.permutation(len(dataset))[:samples]
        return torch.utils.data.Subset(dataset, sample)

    def read_data_mnist(self, trainset=True, samples=5000):
        if trainset:
            dataset = datasets.MNIST(root='./data', train=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,))
            ]), download=True)
            sample = np.random.permutation(len(dataset))[:samples]
        else:
            dataset = datasets.MNIST(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,))
            ]))
            sample = np.random.permutation(len(dataset))[:samples]
        return torch.utils.data.Subset(dataset, sample)

    def read_data_imagenet(self, data_path, trainset=True, samples=5000):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if trainset:
            dataset = ImageNetDataset("/home/chattbap/ILSVRC/train/", transforms.Compose(
                [transforms.RandomSizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]))
            sample = np.random.permutation(len(dataset))[:samples]
        else:
            dataset = ImageNetDataset("/home/chattbap/ILSVRC/val/", transforms.Compose(
                [transforms.RandomSizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]))
            sample = np.random.permutation(len(dataset))[:samples]
        return torch.utils.data.Subset(dataset, sample)
