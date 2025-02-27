import os
import cv2
import numpy as np
import torch
import random
import glob
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
# from datasets.utils import analyze_name
from sklearn.model_selection import KFold
from torchvision.transforms import functional as F
import pandas as pd
import torchvision.transforms as transforms



os.environ["OPENCV_LOG_LEVEL"] = "0"


class Multitask(Dataset):
    def __init__(self, args, split):
        super(Multitask, self).__init__()
        self.args = args
        self.args.size = 1024
        self.args.pseudo_num = 1
        self.args.sub_data = [
            "DRIVE", 
            "FIVES", 
            "HRF", 
            "STARE", 
            # "G1020", 
            # "GAMMA - task3", 
            # "ORIGA", 
            # "Papila", 
            # "REFUGE", 
            # "DDR - lesion_seg", 
            # "FGADR-Seg-set", 
            # "IDRiD"
        ]

        self.x, self.y, self.names = self.load_name(args, split)
        assert len(self.x) == len(self.y) == len(self.names)
        self.dataset_size = len(self.x)
        self.train = True if split == "train" else False
        self.im_transform = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.label_transform = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
            ]
        )
        self.split = split
        self.pseudo_num = self.args.pseudo_num

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        image = cv2.imread(self.x[idx])[..., ::-1]
        image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        # Name
        name = self.names[idx]
        # Label
        mask = self.read_labels(self.y[idx], name, self.split)

        im = Image.fromarray(np.uint8(image))
        mask = Image.fromarray(np.uint8(mask))

        newsize = (1024, 1024)
        mask = mask.resize(newsize)

        y_indices, x_indices = np.where(np.array(mask) > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        H, W = mask.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])

        # Identical transformations for image and ground truth
        seed = np.random.randint(2147483647)
        torch.manual_seed(seed)
        random.seed(seed)
        im_t = self.im_transform(im)
        torch.manual_seed(seed)
        random.seed(seed)
        target_t = self.label_transform(mask)
        torch.manual_seed(seed)
        random.seed(seed)
        image_meta_dict = {'filename_or_obj': name}

        return (
            im_t.float(),
            target_t.long(),
            torch.tensor(bboxes).float(),
            name,
        )

    def read_labels(self, root_dirs, name, split):
        # Read labels for vessel seg
        if root_dirs[0] is not None:
            # print('return label for vessel seg')
            label = Image.open(root_dirs[0])
            label = np.array(label).astype(np.uint8)
            if len(label.shape) == 3:
                label = label[..., 0]
            # label = label[ymin:ymax, xmin:xmax]
            label = cv2.resize(label, (1024, 1024), interpolation=cv2.INTER_NEAREST)

            if not split == 'test':
                # Read pseudo labels for odoc and lesion
                    
                label_pseudo_odoc = cv2.imread(f'/data/wangzh/code/retsam/results_val/index_0/task_1/{name}.png')[..., 0]
                try:
                    label_pseudo_odoc = cv2.resize(label_pseudo_odoc, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                except:
                    print(name)
                # target_pseudo_odoc = Image.fromarray(np.uint8(label_pseudo_odoc))
            
                label_pseudo_lesion = cv2.imread(f'/data/wangzh/code/retsam/results_val/index_0/task_2/{name}.png')[..., 0]
                try:
                    label_pseudo_lesion = cv2.resize(label_pseudo_lesion, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                except:
                    print(name)
                # target_pseudo_lesion = Image.fromarray(np.uint8(label_pseudo_lesion))

                mask = np.zeros_like(label)
                mask[label > 0] = 1
                # mask[label_pseudo_odoc > 1] = 1
                # mask[label_pseudo_odoc == 2] = 2
                # mask[label_pseudo_lesion == 1] = 255
                # mask[label_pseudo_lesion == 2] = 5
                # mask[label_pseudo_lesion == 3] = 6
                # mask[label_pseudo_lesion == 4] = 7

                return mask
            else:
                mask = np.zeros_like(label)
                mask[label > 0] = 1
                return mask

        # Read labels for odoc seg
        elif root_dirs[1] is not None:
            # print('return label for odoc seg')
            label = Image.open(root_dirs[1])
            label = np.array(label).astype(np.uint8)
            if len(label.shape) == 3:
                label = label[..., 0]
            # label = label[ymin:ymax, xmin:xmax]
            label[(label > 0) & (label < 255)] = 1
            label[label == 255] = 2
            label = cv2.resize(label, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            # label = label.resize((1024, 1024))

            # Convert label from numpy to Image
            # target = Image.fromarray(np.uint8(label))

            if not split == 'test':
                # Read pseudo labels for odoc and lesion
                
                label_pseudo_vessel = cv2.imread(f'/data/wangzh/code/retsam/results_val/index_0/task_0/{name}.png')[..., 0]
                label_pseudo_vessel = cv2.resize(label_pseudo_vessel, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                # target_pseudo_vessel = Image.fromarray(np.uint8(label_pseudo_vessel))
            
                label_pseudo_lesion = cv2.imread(f'/data/wangzh/code/retsam/results_val/index_0/task_2/{name}.png')[..., 0]
                label_pseudo_lesion = cv2.resize(label_pseudo_lesion, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                # target_pseudo_lesion = Image.fromarray(np.uint8(label_pseudo_lesion))

                mask = np.zeros_like(label)
                mask[np.where(label > 1)] = 1
                # mask[label == 2] = 2
                # mask[label_pseudo_vessel == 1] = 255
                # mask[label_pseudo_lesion == 1] = 255
                # mask[label_pseudo_lesion == 2] = 5
                # mask[label_pseudo_lesion == 3] = 6
                # mask[label_pseudo_lesion == 4] = 7

                return mask
            else:
                mask = np.zeros_like(label)
                mask[np.where(label > 1)] = 1
                return mask
    
        # Read labels for lesion seg
        else:
            label_ex = Image.open(root_dirs[2]) if root_dirs[2] is not None else np.zeros((1024, 1024), dtype=np.uint8)
            label_he = Image.open(root_dirs[3]) if root_dirs[3] is not None else np.zeros((1024, 1024), dtype=np.uint8)
            label_ma = Image.open(root_dirs[4]) if root_dirs[4] is not None else np.zeros((1024, 1024), dtype=np.uint8)
            label_se = Image.open(root_dirs[5]) if root_dirs[5] is not None else np.zeros((1024, 1024), dtype=np.uint8)

            label_ex = np.array(label_ex).astype(np.uint8)
            label_he = np.array(label_he).astype(np.uint8)
            label_ma = np.array(label_ma).astype(np.uint8)
            label_se = np.array(label_se).astype(np.uint8)

            if len(label_ex.shape) == 3:
                label_ex = label_ex[..., 0]
            if len(label_he.shape) == 3:
                label_he = label_he[..., 0]
            if len(label_ma.shape) == 3:
                label_ma = label_ma[..., 0]
            if len(label_se.shape) == 3:
                label_se = label_se[..., 0]
            
            # try:
            #     label_ex = label_ex[ymin:ymax, xmin:xmax] if root_dirs[2] is not None else np.zeros((1024, 1024), dtype=np.uint8)
            #     label_he = label_he[ymin:ymax, xmin:xmax] if root_dirs[3] is not None else np.zeros((1024, 1024), dtype=np.uint8)
            #     label_ma = label_ma[ymin:ymax, xmin:xmax] if root_dirs[4] is not None else np.zeros((1024, 1024), dtype=np.uint8)
            #     label_se = label_se[ymin:ymax, xmin:xmax] if root_dirs[5] is not None else np.zeros((1024, 1024), dtype=np.uint8)
            # except:
            #     print(root_dirs)
            
            label_ex = cv2.resize(label_ex, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            label_he = cv2.resize(label_he, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            label_ma = cv2.resize(label_ma, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            label_se = cv2.resize(label_se, (1024, 1024), interpolation=cv2.INTER_NEAREST)

            label = np.zeros((label_ex.shape[0], label_ex.shape[1]), dtype=np.uint8)
            label[np.where(label_ex == 255)] = 1
            label[np.where(label_he == 255)] = 2
            label[np.where(label_ma == 255)] = 3
            label[np.where(label_se == 255)] = 4

            label = cv2.resize(label, (1024, 1024), interpolation=cv2.INTER_NEAREST)

            # Convert label from numpy to Image
            # target = Image.fromarray(np.uint8(label))

            if not split == 'test':
                # Read pseudo labels for vessel and odoc
                label_pseudo_vessel = cv2.imread(f'/data/wangzh/code/retsam/results_val/index_0/task_0/{name}.png')[..., 0]
                label_pseudo_vessel = cv2.resize(label_pseudo_vessel, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                # target_pseudo_vessel = Image.fromarray(np.uint8(label_pseudo_vessel))
            
                label_pseudo_odoc = cv2.imread(f'/data/wangzh/code/retsam/results_val/index_0/task_1/{name}.png')[..., 0]
                label_pseudo_odoc = cv2.resize(label_pseudo_odoc, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                # target_pseudo_odoc = Image.fromarray(np.uint8(label_pseudo_odoc))

                mask = np.zeros_like(label)
                mask[label == 1] = 1
                # mask[label == 2] = 5
                # mask[label == 3] = 6
                # mask[label == 4] = 7
                # mask[label_pseudo_vessel == 1] = 255
                # mask[label_pseudo_odoc > 1] = 1
                # mask[label_pseudo_odoc == 2] = 2

                return mask
            else:
                mask = np.zeros_like(label)
                mask[label == 1] = 1
                return mask

    def read_images(self, root_dir):
        image_paths = []
        patterns = ['*.png', '*.jpg', '*.jpeg']
        for pattern in patterns:
            full_pattern = os.path.join(root_dir, '**', pattern)
            found_paths = glob.glob(full_pattern, recursive=True)
            image_paths.extend(found_paths)
        return image_paths

    def load_name(self, args, split):
        inputs, targets, names = [], [], []

        vessel_inputs, vessel_targets, vessel_names = self.load_vessel(args, split)
        ODOC_inputs, ODOC_targets, ODOC_names = self.load_ODOC(args, split)
        lesion_inputs, lesion_targets, lesion_names = self.load_lesion(args, split)

        for i in range(len(vessel_inputs)):
            inputs.append(vessel_inputs[i])
            targets.append([vessel_targets[i], None, None, None, None, None])
            names.append(vessel_names[i])
        
        for i in range(len(ODOC_inputs)):
            inputs.append(ODOC_inputs[i])
            targets.append([None, ODOC_targets[i], None, None, None, None])
            names.append(ODOC_names[i])

        for i in range(len(lesion_inputs)):
            inputs.append(lesion_inputs[i])
            targets.append([None, None, lesion_targets[0][i], lesion_targets[1][i], lesion_targets[2][i], lesion_targets[3][i]])
            names.append(lesion_names[i])
        
        inputs = np.array(inputs)
        targets = np.array(targets)
        names = np.array(names)

        return inputs, targets, names

    def load_vessel(self, args, split):
        inputs, targets, names = [], [], []

        # Define root directories
        root_dir = '/data/wangzh/datasets/fundus_miccai/vessel_seg/'

        for dataset in args.sub_data:
            csv_path = os.path.join(root_dir, f'{dataset}/{split}.csv')

            if not os.path.exists(csv_path):
                continue
            data = pd.read_csv(csv_path)
            images = data['image_path'].values
            labels = data['label_path'].values
            image_names = data['image_name'].values
            image_names = [dataset + '_' + split + '_' + str(name) for name in image_names]

            inputs.extend(images)
            targets.extend(labels)
            names.extend(image_names)

        inputs = [str.replace('/datasets/vessel_seg/', root_dir) for str in inputs]
        targets = [str.replace('/datasets/vessel_seg/', root_dir) for str in targets]

        inputs = np.array(inputs)
        targets = np.array(targets)
        names = np.array(names)

        print("=> Using {} images for vessel {}".format(len(inputs), split))
        return inputs, targets, names

    def load_ODOC(self, args, split):
        inputs, targets, names = [], [], []

        # Define root directories
        root_dir = '/data/wangzh/datasets/fundus_miccai/ODOC_seg/'

        for dataset in args.sub_data:
            csv_path = os.path.join(root_dir, f'{dataset}/{split}.csv')

            if not os.path.exists(csv_path):
                continue
            data = pd.read_csv(csv_path)
            images = data['image_path'].values
            labels = data['label_path'].values
            image_names = data['image_name'].values
            image_names = [dataset + '_' + split + '_' + str(name) for name in image_names]

            inputs.extend(images)
            targets.extend(labels)
            names.extend(image_names)

        inputs = [str.replace('/datasets/ODOC_seg/', root_dir) for str in inputs]
        targets = [str.replace('/datasets/ODOC_seg/', root_dir) for str in targets]

        inputs = np.array(inputs)
        targets = np.array(targets)
        names = np.array(names)

        print("=> Using {} images for ODOC {}".format(len(inputs), split))
        return inputs, targets, names

    def load_lesion(self, args, split):
        inputs, targets_ex, targets_he, targets_ma, targets_se, names = [], [], [], [], [], []

        # Define root directories
        root_dir = '/data/wangzh/datasets/fundus_miccai/lesion_seg/'

        for dataset in args.sub_data:
            csv_path = os.path.join(root_dir, f'{dataset}/{split}.csv')

            if not os.path.exists(csv_path):
                continue
            data = pd.read_csv(csv_path)
            images = data['image_path'].values
            for col in ["label_EX_path", "label_HE_path", "label_MA_path", "label_SE_path"]:
                data[col] = data[col].apply(lambda x: None if pd.isna(x) else x)
            labels_ex = data['label_EX_path'].values
            labels_he = data['label_HE_path'].values
            labels_ma = data['label_MA_path'].values
            labels_se = data['label_SE_path'].values
            image_names = data['image_name'].values
            image_names = [dataset + '_' + split + '_' + str(name) for name in image_names]

            inputs.extend(images)
            targets_ex.extend(labels_ex)
            targets_he.extend(labels_he)
            targets_ma.extend(labels_ma)
            targets_se.extend(labels_se)
            names.extend(image_names)

        for i in range(len(inputs)):
            inputs[i] = str.replace(inputs[i], '/datasets/lesion_seg/', root_dir)
            if targets_ex[i] is not None:
                targets_ex[i] = str.replace(targets_ex[i], '/datasets/lesion_seg/', root_dir)
            if targets_he[i] is not None:
                targets_he[i] = str.replace(targets_he[i], '/datasets/lesion_seg/', root_dir)
            if targets_ma[i] is not None:
                targets_ma[i] = str.replace(targets_ma[i], '/datasets/lesion_seg/', root_dir)
            if targets_se[i] is not None:
                targets_se[i] = str.replace(targets_se[i], '/datasets/lesion_seg/', root_dir)

        inputs = np.array(inputs)
        targets_ex = np.array(targets_ex)
        targets_he = np.array(targets_he)
        targets_ma = np.array(targets_ma)
        targets_se = np.array(targets_se)
        names = np.array(names)

        print("=> Using {} images for Lesion {}".format(len(inputs), split))
        return inputs, [targets_ex, targets_he, targets_ma, targets_se], names


def load_dataset(args, train=False):
    if train:
        train_dataset = Multitask(args, 'train')
        val_dataset = Multitask(args, 'val')
        return train_dataset, val_dataset
    else:
        test_dataset = Multitask(args, 'test')
        return test_dataset
