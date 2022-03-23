from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.utils.data import Dataset
import os
import re
import torch
from copy import deepcopy
from collections import defaultdict
from PIL import Image
import torchvision.transforms as transforms

IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg']


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def find_images_and_targets(folder, types=IMG_EXTENSIONS, leaf_name_only=True, sort=True):
    labels = []
    filenames = []
    for root, subdirs, files in os.walk(folder, topdown=False):
        rel_path = os.path.relpath(root, folder) if (root != folder) else ''
        label = os.path.basename(rel_path) if leaf_name_only else rel_path.replace(os.path.sep, '_')
        for f in files:
            base, ext = os.path.splitext(f)
            if ext.lower() in types:
                filenames.append(os.path.join(root, f))
                labels.append(label)
    unique_labels = set(labels)
    # for i in range(len(unique_labels), 1000):
    #     unique_labels.add("n" + str(i))
    sorted_labels = list(sorted(unique_labels, key=natural_key))
    class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
    images_and_targets = zip(filenames, [class_to_idx[l] for l in labels])
    if sort:
        images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
    return images_and_targets, class_to_idx


class ImageNetDataset(Dataset):
    def __init__(self, root, transform=None):
        images, class_to_idx = find_images_and_targets(root)
        if len(images) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n" + "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.class_to_idx = class_to_idx
        self.root = root
        self.samples = images
        self.imgs = self.samples  # torchvision ImageFolder compat
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            img = Image.open(path).convert('RGB')
            # print(img.shape)
            # Image.open(path).convert('RGB')
        except:
            print(f'Warning : failed to loader {path}')
            img = Image.new('RGB', (224, 224))
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = torch.zeros(1).long()
        return img, target

    def __len__(self):
        return len(self.imgs)

    # def filenames(self, indices=[], basename=False):
    #     if indices:
    #         if basename:
    #             return [os.path.basename(self.samples[i][0]) for i in indices]
    #         else:
    #             return [self.samples[i][0] for i in indices]
    #     else:
    #         if basename:
    #             return [os.path.basename(x[0]) for x in self.samples]
    #         else:
    #             return [x[0] for x in self.samples]
    #
    # def split_dataset(self, split_ratio, **kwargs):
    #     dataset1 = deepcopy(self)
    #     dataset2 = deepcopy(self)
    #     target_dict = defaultdict(list)
    #     for image, target in self.samples:
    #         target_dict[target].append(image)
    #     dataset1.samples = []
    #     dataset2.samples = []
    #     for k, v in target_dict.items():
    #         num_elem1 = int(len(v) * split_ratio)
    #         num_elem2 = len(v) - num_elem1
    #         dataset1.samples.extend(zip(v[:num_elem1], [k] * num_elem1))
    #         dataset2.samples.extend(zip(v[num_elem1:], [k] * num_elem2))
    #     dataset1.imgs = dataset1.samples
    #     dataset2.imgs = dataset2.samples
    #     return dataset1, dataset2


if __name__ == "__main__":                                     
    train = ImageNetDataset("/home/chattbap/ILSVRC/train/", transforms.Compose([transforms.RandomSizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]))
    test = ImageNetDataset("/home/chattbap/ILSVRC/val/", transforms.Compose([transforms.RandomSizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]))
    if train.class_to_idx == test.class_to_idx:
        print(True)