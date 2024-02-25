import os
import torch.utils.data as data
from .dataset import IncrementalSegmentationDataset
import numpy as np
from xml.etree import ElementTree
import zipfile
from PIL import Image, ImageDraw
from .image_helper import ImageHelper
import torch


classes = {
    0: 'background',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor'
}
task_list = ['person', 'animals', 'vehicles', 'indoor']
tasks = {
    'person': [15],
    'animals': [3, 8, 10, 12, 13, 17],
    'vehicles': [1, 2, 4, 6, 7, 14, 19],
    'indoor': [5, 9, 11, 16, 18, 20]
}

coco_map = [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]


class VOCSegmentation(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        is_aug (bool, optional): If you want to use the augmented train set or not (default is True)
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 indices=None,
                 as_coco=False,
                 saliency=False,
                 pseudo=None):

        self.root = os.path.expanduser(root)
        self.year = "2012"

        self.ds_scribbles_path = './dataset/voc_scribbles.zip'
        assert os.path.isfile(self.ds_scribbles_path), f'Scribbles not found at {self.ds_scribbles_path}'
        self.integrity_check = True
        self.stroke_width = 3
        self.semseg_ignore_class = 255

        self.cls_name_to_id = {name: i for i, name in enumerate(self.semseg_class_names)}

        self.transform = transform

        self.image_set = 'train' if train else 'val'
        base_dir = "voc"
        voc_root = os.path.join(self.root, base_dir)
        splits_dir = os.path.join(voc_root, 'splits')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' Download it')

        mask_dir = os.path.join(voc_root, 'SegmentationClassAug')
        assert os.path.exists(mask_dir), "SegmentationClassAug not found"

        if as_coco:
            if train:
                split_f = os.path.join(splits_dir, 'train_aug_ascoco.txt')
            else:
                split_f = os.path.join(splits_dir, 'val_ascoco.txt')
        else:
            if train:
                split_f = os.path.join(splits_dir, 'train_aug.txt')
            else:
                split_f = os.path.join(splits_dir, 'val.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        # remove leading \n
        with open(os.path.join(split_f), "r") as f:
            file_names = [x[:-1].split(' ') for x in f.readlines()]

        # REMOVE FIRST SLASH OTHERWISE THE JOIN WILL start from root
        self.images = [(os.path.join(voc_root, x[0][1:]), os.path.join(voc_root, x[1][1:])) for x in file_names]
        if saliency:
            self.saliency_images = [x[0].replace("JPEGImages", "SALImages")[:-3] + "png" for x in self.images]
        else:
            self.saliency_images = None

        # change ground truth annotation to pseudo label
        if pseudo is not None and train:
            if not as_coco:
                self.images = [(x[0], x[1].replace("SegmentationClassAug", f"PseudoLabels/{pseudo}/rw/")) for x in self.images]
            else:
                self.images = [(x[0], x[1].replace("SegmentationClassAugAsCoco", f"PseudoLabels/{pseudo}/rw")) for x in
                               self.images]
        if as_coco:
            self.img_lvl_labels = np.load(os.path.join(voc_root, f"cocovoc_1h_labels_{self.image_set}.npy"))
        else:
            self.img_lvl_labels = np.load(os.path.join(voc_root, f"voc_1h_labels_{self.image_set}.npy"))

        self.indices = indices if indices is not None else np.arange(len(self.images))

    def _parse_scribble(self, name, known_width=None, known_height=None):
        with zipfile.ZipFile(self.ds_scribbles_path, 'r') as f:
            data = f.read(name + '.xml')
        sample_xml = ElementTree.fromstring(data)
        assert sample_xml.tag == 'annotation', f'XML error in sample {name}'
        found_size = False
        polylines = []
        for i in range(len(sample_xml)):
            if sample_xml[i].tag == 'size':
                found_size = True
                found_width, found_height = False, False
                sample_xml_size = sample_xml[i]
                for j in range(len(sample_xml_size)):
                    if sample_xml_size[j].tag == 'width':
                        assert known_width is None or int(sample_xml_size[j].text) == known_width, \
                            f'XML error in sample {name}'
                        found_width = True
                    elif sample_xml_size[j].tag == 'height':
                        assert known_height is None or int(sample_xml_size[j].text) == known_height, \
                            f'XML error in sample {name}'
                        found_height = True
                assert found_width and found_height, f'XML error in sample {name}'
            if sample_xml[i].tag == 'polygon':
                polygon = sample_xml[i]
                polygon_class, polygon_points = None, []
                for j in range(len(polygon)):
                    polygon_entry = polygon[j]
                    if polygon_entry.tag == 'tag':
                        polygon_class = polygon_entry.text
                    elif polygon_entry.tag == 'point':
                        assert polygon_entry[0].tag == 'X' and polygon_entry[1].tag == 'Y', \
                            f'XML error in sample {name}'
                        polygon_points.append((int(polygon_entry[0].text), int(polygon_entry[1].text)))
                assert polygon_class is not None and len(polygon_points) > 0, f'XML error in sample {name}'
                polylines.append((self.cls_name_to_id[polygon_class], polygon_points))
        assert found_size and len(polylines) > 0, f'XML error in sample {name}'
        # coordinate tuples have (x,y) order
        return polylines

    def rasterize_scribbles(self, data, width, height):
        img = Image.new("L", (width, height), color=self.semseg_ignore_class)
        draw = ImageDraw.Draw(img)
        polylines = data
        for clsid, joints in polylines:
            if len(joints) > 1:
                draw.line(joints, clsid, self.stroke_width, joint="curve")
            for i in range(len(joints)):
                draw.ellipse((
                    joints[i][0] - self.stroke_width / 2, joints[i][1] - self.stroke_width / 2,
                    joints[i][0] + self.stroke_width / 2, joints[i][1] + self.stroke_width / 2),
                    clsid
                )
        return img

    from PIL import Image, ImageOps


    @staticmethod
    def _sample_name(path):
        return path.split('/')[-1].split('.')[0]

    @property
    def semseg_class_names(self):
        return [
            'background',
            'plane',
            'bike',
            'bird',
            'boat',
            'bottle',
            'bus',
            'car',
            'cat',
            'chair',
            'cow',
            'table',
            'dog',
            'horse',
            'motorbike',
            'person',
            'plant',
            'sheep',
            'sofa',
            'train',
            'monitor',
        ]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[self.indices[index]][0]).convert('RGB')
        target = Image.open(self.images[self.indices[index]][1])
        img_lvl_lbls = self.img_lvl_labels[self.indices[index]]

        width, height = ImageHelper.get_size(img)
        # print("width ", width)
        # print("height ", height)
        # img_lvl_lbls = self.img_lvl_labels[self.indices[index]]
        scribble_name = self._sample_name(self.images[self.indices[index]][0])

        scribble_data = self._parse_scribble(scribble_name, width, height)
        scrib_lbl = self.rasterize_scribbles(scribble_data, width, height)
        # return img, target, img_lvl_lbls

        if self.transform is not None:
            img, target = self.transform(img, target, scrib_lbl)


        # print(type(img))
        # print(type(target))

        # gets called first
        return img, target, scrib_lbl, img_lvl_lbls

    def __len__(self):
        return len(self.indices)


class VOCSegmentationIncremental(IncrementalSegmentationDataset):
    def make_dataset(self, root, train, indices, saliency=False, pseudo=None):
        full_voc = VOCSegmentation(root, train, transform=None, indices=indices, saliency=saliency, pseudo=pseudo)
        return full_voc


class VOCasCOCOSegmentationIncremental(IncrementalSegmentationDataset):
    def make_dataset(self, root, train, indices, saliency=False, pseudo=None):
        full_voc = VOCSegmentation(root, train, transform=None, indices=indices, as_coco=True,
                                   saliency=saliency, pseudo=pseudo)
        return full_voc


class LabelTransform:
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, x):
        return Image.fromarray(self.mapping[x])
