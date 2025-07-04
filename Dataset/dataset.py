import logging

import os
import cv2

from torch.utils.data import Dataset

from Dataset.test import mixup_bboxes


# from Dataset.test import transformed


class Path22Dataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.label_file = label_file
        self.transform = transform
        self.images = []
        self.labels = {}
        self._load_data()

    def _load_data(self):
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory {self.image_dir} does not exist.")
        if not os.path.exists(self.label_file):
            raise FileNotFoundError(f"Label file {self.label_file} does not exist.")
        self.images = os.listdir(self.image_dir)
        # for image in self.images:
        labels = list(open(self.label_file, 'r').read().splitlines())
        labels = [[int(x) for x in line.split()] for line in labels]

        for image in self.images:
            image_id = int(image.split('.')[0])
            image_path = os.path.join(self.image_dir, image)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height_img, wight_img = img.shape[0:2]
            self.labels[image] = [label[2:6] + [0] for label in
                                  list(filter(lambda x: int(x[0]) == image_id, labels))]
            self.labels[image] = [
                [(label[0] + label[2] / 2) / wight_img, (label[1] + label[3] / 2) / height_img,
                 label[2] / wight_img, label[3] / height_img, label[4]]  # Convert to normalized coordinates
                for label in self.labels[image]
            ]  # x y w h C

    def __getitem__(self, index):
        image_name = self.images[index]
        image_path = os.path.join(self.image_dir, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = Image.open(image_path).convert('RGB')
        label = self.labels[image_name]
        if self.transform:
            image = self.transform(image=image, label=label)['image']
        return image, self.labels[image_name]

    def __len__(self):
        return len(self.images)

    def changeFomat(self, folder_dir='Datatest', new_name='test', image_format='.jpg', label_format='.txt',
                    make_dir=True):
        if make_dir:
            os.makedirs(folder_dir, exist_ok=True)
            os.makedirs(os.path.join(folder_dir, 'image'), exist_ok=True)
            os.makedirs(os.path.join(folder_dir, 'label'), exist_ok=True)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s')
        for image_name in self.images:
            image_path = os.path.join(self.image_dir, image_name)
            image = cv2.imread(image_path)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            labels = self.labels[image_name]
            new_image_name = new_name + "_" + image_name.split('.')[0]
            with open(os.path.join(folder_dir, 'label', f"{new_image_name.split('.')[0]}{label_format}"), 'x') as f:
                for label in labels:
                    x, y, w, h, C = label
                    f.write(f"{C} {x} {y} {w} {h} \n")
            cv2.imwrite(os.path.join(folder_dir, 'image', new_image_name + image_format), image)
            logging.info(f" completed {new_image_name}{image_format} and {new_image_name.split('.')[0]}{label_format} ")


class DataSet(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, anchor_boxes=None):
        """
        Custom dataset for loading images and their corresponding labels.
        :param image_dir:
        :param label_dir:
        :param transform:
        :param anchor_boxes:
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.labels = os.listdir(label_dir)
        if anchor_boxes is not None:
            self.anchor_boxes = anchor_boxes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index]
        label_name = self.images[index].split('.')[0] + '.txt'
        image_path = os.path.join(self.image_dir, image_name)
        label_path = os.path.join(self.label_dir, label_name)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = Image.open(image_path).convert('RGB')
        with open(label_path, 'r') as f:
            labels = f.readlines()
        labels = [list(map(float, line.strip().split())) for line in labels]
        labels = [[max(0.0, min(1.0, label[1])),
                   max(0.0, min(1.0, label[2])),
                   max(0.0, min(1.0, label[3])),
                   max(0.0, min(1.0, label[4])), int(label[0])] for label in
                  labels]  # Convert to [x, y, w, h, class]
        # labels = np.array(labels)
        if self.transform:
            image = self.transform(image=image, bboxes=labels)['image']
        return image, labels


def changeFomat(forder_dir, new_folder_dir='test', image_format='.jpg', label_format='.txt', make_dir=True):
    list_dataset = os.listdir(forder_dir)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s')
    for image_idr in list_dataset:
        image_dir = os.path.join(forder_dir, image_idr, 'img1')
        label_dir = os.path.join(forder_dir, image_idr, 'gt', 'gt.txt')
        dataset = Path22Dataset(image_dir=image_dir, label_file=label_dir)
        dataset.changeFomat(folder_dir=os.path.join(new_folder_dir), make_dir=make_dir, new_name=image_idr,
                            image_format=image_format, label_format=label_format)
        logging.info(f"Completed processing {image_idr} dataset.")


def main():
    import matplotlib.pyplot as plt
    from utills.drawBoundingBoxes import drawBboxes
    from utills.MixupAugmentation import MixupAugmentationV2
    import albumentations as A

    imgage_dir = 'test/image'
    label_dir = 'test/label'
    dataset = DataSet(image_dir=imgage_dir, label_dir=label_dir)

    image1 = dataset[100][0]
    bboxes1 = dataset[100][1]
    image2 = dataset[10][0]
    bboxes2 = dataset[10][1]

    transform = A.Compose([
        MixupAugmentationV2(mixup_alpha=0.9, p=1.0),
        # A.Resize(640, 640),
    ],
        # bbox_params=A.BboxParams(format="yolo", clip=True)
    )

    # tran = DataAugmentation(image=image1, bboxes=label1)
    tran = transform(image=image1, bboxes=bboxes1, mixup_image=image2, mixup_bboxes=bboxes2)
    # trans = MixupAugmentationV2(mixup_alpha=0.9, p=1.0)
    # tran = trans(image=image1, bboxes=bboxes1, mixup_image=image2, mixup_bboxes=bboxes2)

    cv2.imshow("img1", drawBboxes(image1, bboxes1))
    cv2.imshow("img2", drawBboxes(image2, bboxes2))
    cv2.imshow('image', drawBboxes(tran['image'], tran['bboxes']))
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
    pass
