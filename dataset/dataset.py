import os
import torch.utils.data as data
from torch import from_numpy
import numpy as np


class IncrementalSegmentationDataset(data.Dataset):
    def __init__(self,
                 root,
                 step_dict,
                 train=True,
                 transform=None,
                 idxs_path=None,
                 masking=True,
                 overlap=True,
                 masking_value=0,
                 step=0,
                 weakly=False,
                 pseudo=None):

        # Load the indices of images based on the training split.
        # take index of images with at least one class in labels and all classes in labels+labels_old+[255]
        if train:
            if idxs_path is not None and os.path.exists(idxs_path):
                idxs = np.load(idxs_path)
            else:
                raise FileNotFoundError(f"Please, add the traning spilt in {idxs_path}.")
        else:  # In both test and validation we want to use all data available (even if some images are all bkg)
            idxs = None

        # Create the dataset using a subclass's implementation of make_dataset.
        self.dataset = self.make_dataset(root, train, indices=idxs, pseudo=pseudo)
        self.transform = transform
        self.weakly = weakly  # Indicates whether weakly supervised learning is used (not for validation)
        self.train = train # Indicates training or validation/testing mode

        # step_dict: A dictionary defining which classes are introduced at each step of training
        self.step_dict = step_dict
        self.labels = [] # Current set of classes for this step
        self.labels_old = [] # Classes from previous steps
        self.step = step # Current step in the continual learning process

        # order: A list of classes in the order they are introduced
        self.order = [c for s in sorted(step_dict) for c in step_dict[s]]
        # assert not any(l in labels_old for l in labels), "Labels and labels_old must be disjoint sets"
        # print("step", step)
        # print("weakly", weakly)
        # print("train", train)
        # print("step_dict", step_dict)
        # print("self.order", self.order)
        # Set up current and old labels based on the step
        if step > 0:
            self.labels = [self.order[0]] + list(step_dict[step])
        else:
            self.labels = list(step_dict[step])
        self.labels_old = [lbl for s in range(step) for lbl in step_dict[s]]
        # print("self.labels", self.labels)
        # Masking for handling classes that are not yet introduced or should be ignored
        self.masking_value = masking_value
        self.masking = masking

        # inverted_order: Maps each class to its index in the order list
        self.inverted_order = {lb: self.order.index(lb) for lb in self.order}
        if train:
            self.inverted_order[255] = masking_value
        else:
            self.set_up_void_test()

        if masking:
            tmp_labels = self.labels + [255]
            mapping_dict = {x: self.inverted_order[x] for x in tmp_labels}
        else:
            mapping_dict = self.inverted_order

        # mapping: An array used to remap label values for the LabelTransform
        # if not (train and self.weakly):
        mapping = np.zeros((256,))
        for k in mapping_dict.keys():
            mapping[k] = mapping_dict[k]

        self.transform_lbl = LabelTransform(mapping) # Transformation for remapping label values

        # LabelSelection: Used for handling label selection based on the current order and active labels, not used in scribble
        self.transform_1h = LabelSelection(self.order, self.labels, self.masking)

    def set_up_void_test(self):
        self.inverted_order[255] = 255

    def __getitem__(self, index):
        # Data loading for a given index. Applies transformations to image, label, and scribble
        if index < 0:
            if -index > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            index = len(self) + index

        if index < len(self):
            data = self.dataset[index]
            img, lbl, scribble_label, lbl_1h = data[0], data[1], data[2], data[3]
            # print("Before")
            # print("lbl_1h: ", lbl_1h.shape)
            # print("after")
            img, lbl, scribble_label = self.transform(img, lbl, scribble_label)
            lbl = self.transform_lbl(lbl) # Apply label transform to both label and scribble
            scribble_label = self.transform_lbl(scribble_label)
            lbl_1h = self.transform_1h(lbl_1h)
            # print("img: ", img.shape)
            # print("lbl: ", lbl.shape)
            # print("lbl_1h: ", scribble_label.shape)

            # gets called second
            
            # print(img.shape)
            # print(lbl.shape)
            # print(scribble_label.shape)
            # print(lbl_1h.shape)
            return img, lbl, scribble_label, lbl_1h

        else:
            raise ValueError("absolute value of index should not exceed dataset length")

    @staticmethod
    def __strip_zero(labels):
        while 0 in labels:
            labels.remove(0)

    def __len__(self):
        return len(self.dataset)

    def make_dataset(self, root, train, indices, saliency=False, pseudo=None):
        raise NotImplementedError

# Does mapping
class LabelTransform:
    # Transform for remapping label values based on a given mapping
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, x):
        return from_numpy(self.mapping[x])


class LabelSelection:
    def __init__(self, order, labels, masking):
        order = np.array(order)
        order = order[order != 0]
        order -= 1  # scale to match one-hot index.
        self.order = order
        if masking:
            self.masker = np.zeros((len(order)))
            self.masker[-len(labels)+1:] = 1
        else:
            self.masker = np.ones((len(order)))

    def __call__(self, x):
        x = x[self.order] * self.masker
        return x
