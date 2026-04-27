import copy
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class MyDatasetLoader(Dataset):
    def __init__(self, data_npy, mode, pwd=None, transform=None, device="cuda:0"):
        super(MyDatasetLoader, self).__init__()

        imgs_dir = np.load(data_npy, allow_pickle=True)

        if pwd:
            self.pwd = pwd
        else:
            self.pwd = self.get_pwd(imgs_dir, data_npy)

        self.device = device
        self.mode = mode

        dataset_num = len(imgs_dir)
        self.dataset = imgs_dir

        self.transform = transform

    # def __getitem__(self, item):
    #     data_dict = self.dataset[item]
    #     imageData, maskData, class_label = self.rDatadict(data_dict)
    #
    #     imageData, maskData = self.imagepreprocess(imageData, maskData)
    #     imageData, maskData = self.augmentation(imageData, maskData)
    #
    #     class_label = class_label.split('-')[0]
    #     class_label = int(class_label)
    #
    #     return imageData, maskData, class_label

    def __getitem__(self, item):
        data_dict = self.dataset[item]
        # Returns patient_id and filename in addition to image/mask/label
        imageData, maskData, class_label, patient_name, file_name = self.rDatadict(data_dict)

        ori_imageData = copy.deepcopy(imageData)
        data_shape = np.shape(imageData)

        imageData, maskData = self.imagepreprocess(imageData, maskData)
        imageData, maskData = self.augmentation(imageData, maskData)

        class_label = class_label.split('-')[0]  # e.g. "1-Normal" -> "1"
        class_label = int(class_label)

        if self.mode == "test":
            return imageData, maskData, class_label, patient_name, file_name, data_shape, ori_imageData
        else:
            return imageData, maskData, class_label, patient_name

    def __len__(self):
        return len(self.dataset)

    def get_pwd(self, imgs_dir, data_npy):
        example = imgs_dir[0]
        image_path = example["SWEroi2"]
        dir = image_path.split('/')[0]
        pwd = data_npy.split(dir)[0]
        return pwd

    # def rDatadict(self, dict):
    #     ## Read image data
    #     image_path = dict[0]
    #     class_label = dict[1]
    #     file_name = image_path.split('/')[-2]
    #     image = Image.open(image_path).convert('L')
    #     image = np.asarray(image)
    #
    #     ## Read clinical data
    #     mask_path = image_path.replace("_image", "_mask")
    #     mask = Image.open(mask_path).convert('L')
    #     mask = np.asarray(mask)
    #
    #     return image, mask, class_label

    def rDatadict(self, dict):
        image_path = dict[0]  # e.g. "subject_case001_image.png"
        class_label = dict[1]  # folder tag, e.g. "1-Normal"

        full_image_path = os.path.join(self.pwd, class_label, image_path)

        file_name = os.path.basename(full_image_path)
        if "_image." in file_name:
            patient_name = file_name.split("_image.")[0]
        else:
            patient_name = file_name.split("_")[0]

        image = Image.open(full_image_path).convert('L')
        image = np.asarray(image)
        mask_path = full_image_path.replace("_image", "_mask")
        mask = Image.open(mask_path).convert('L')
        mask = np.asarray(mask)

        return image, mask, class_label, patient_name, file_name

    def imagepreprocess(self, image, mask):
        if len(image.shape) < 3:
            image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, 0)
        image = np.transpose(image, [0, 3, 1, 2])

        image = image.astype('float32')

        # image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = image / 255
        # image = (image - np.mean(image)) / np.std(image)

        if len(mask.shape) < 3:
            mask = np.expand_dims(mask, axis=-1)
        mask = np.expand_dims(mask, 0)
        mask = np.transpose(mask, [0, 3, 1, 2])

        mask = mask.astype('float32')
        mn, mx = float(np.min(mask)), float(np.max(mask))
        if mx - mn < 1e-8:
            mask = np.zeros_like(mask)
        else:
            mask = (mask - mn) / (mx - mn)
        mask = mask * 2
        return image, mask

    def augmentation(self, image, mask):
        data_dict = dict(data=image, seg=mask)

        if self.transform is not None:
            augmented = self.transform(**data_dict)
            image = augmented.get("data")
            mask = augmented.get("seg")
        image = image[0, :, :, :]
        mask = mask[0, :, :, :]
        # After geometric aug + interpolation, re-quantize labels to {0,1,2} for CE targets
        mask = np.clip(np.round(mask), 0, 2).astype(np.float32)
        return image, mask
