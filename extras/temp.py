# import os
# import re
# import shutil
#
# IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg']
#
#
# def natural_key(string_):
#     """See http://www.codinghorror.com/blog/archives/001018.html"""
#     return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]
#
#
# def find_images_and_targets(folder, types=IMG_EXTENSIONS, class_to_idx=None, leaf_name_only=True, sort=True):
#     labels = []
#     filenames = []
#     for root, subdirs, files in os.walk(folder, topdown=False):
#         # print(root, subdirs, files)
#         rel_path = os.path.relpath(root, folder) if (root != folder) else ''
#         # print(rel_path)
#         label = os.path.basename(rel_path) if leaf_name_only else rel_path.replace(os.path.sep, '_')
#         # label = rel_path.split(os.path.sep)[-2] if leaf_name_only else rel_path.replace(os.path.sep, '_')
#         print(label)
#         for f in files:
#             base, ext = os.path.splitext(f)
#             if ext.lower() in types:
#                 filenames.append(os.path.join(root, f))
#                 labels.append(label)
#     if class_to_idx is None:
#         # building class index
#         unique_labels = set(labels)
#         sorted_labels = list(sorted(unique_labels, key=natural_key))
#
#         class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
#     images_and_targets = zip(filenames, [class_to_idx[l] for l in labels])
#     if sort:
#         images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
#     return images_and_targets, class_to_idx
#
# def extract_images(folder, types=IMG_EXTENSIONS, class_to_idx=None, leaf_name_only=True, sort=True):
#     labels = []
#     filenames = []
#     for root, subdirs, files in os.walk(folder, topdown=False):
#         # print(root, subdirs, files)
#         rel_path = os.path.relpath(root, folder) if (root != folder) else ''
#         # print(rel_path)
#         # label = os.path.basename(rel_path)  if leaf_name_only else rel_path.replace(os.path.sep, '_')
#         # label = rel_path.split(os.path.sep)[-2] if leaf_name_only else rel_path.replace(os.path.sep, '_')
#         # print(label)
#         if os.path.basename(rel_path) == "images":
#             for f in files:
#                 file_name = os.path.join(root, f)
#                 # print(rel_path)
#                 # print()
#                 # print(os.path.join(root, os.path.normpath(rel_path + os.sep + os.pardir)))
#                 # print(file_name)
#                 shutil.move(file_name, os.path.normpath(root + os.sep + os.pardir))
#                 # base, ext = os.path.splitext(f)
#                 # if ext.lower() in types:
#                 #     filenames.append(os.path.join(root, f))
#                 #     labels.append(label)
#
#
# find_images_and_targets("../data/Imagenet/train")
# # extract_images("../data/Imagenet/train")
#
#
import collections

import numpy as np

# b = dict(np.load(, allow_pickle=True))
# print(b.keys())
# print( ("momentum_change" in b.keys()))
import os
import sys

sys.path.append(os.getcwd())
from helper.pytorch.pytorch_helper import PytorchHelper
from model.pytorch_model_trainer import np_to_grad, np_to_weights

helper = PytorchHelper()
model, momentum, momentum_change = helper.load_model("initial_model.npz")
global_momentum = np_to_grad(momentum)
for change in global_momentum:
    for key in change.keys():
        print(change[key].shape)
momentum_change = np_to_grad(momentum_change)
for change in momentum_change:
    for key in change.keys():
        print(change[key].shape)
weights_np = np_to_weights(model)
print(weights_np.keys())