import torch
import os
import cv2
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A


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
            # self.labels[image] = [[0] + label[2:6] for label in list(filter(lambda x: int(x[0]) == image_id, labels))]
            # self.labels[image] = [
            #     [label[0], (label[1] + label[3] / 2) / wight_img, (label[2] + label[4] / 2) / height_img,
            #      label[3] / wight_img, label[4] / height_img]  # Convert to normalized coordinates
            #     for label in self.labels[image]
            # ]
            self.labels[image] = [label[2:6] + [0] for label in
                                  list(filter(lambda x: int(x[0]) == image_id, labels))]
            self.labels[image] = [
                [(label[0] + label[2] / 2) / wight_img, (label[1] + label[3] / 2) / height_img,
                 label[2] / wight_img, label[3] / height_img, label[4]]  # Convert to normalized coordinates
                for label in self.labels[image]
            ]

    def __getitem__(self, index):
        image_name = self.images[index]
        image_path = os.path.join(self.image_dir, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = Image.open(image_path).convert('RGB')
        label = self.labels[image_name]
        if self.transform:
            image = self.transform(image=image, label=label)['image']
        return image, self.labels[image_name]

    def __len__(self):
        return len(self.images)


# transform = A.Compose([
#     A.Mosaic(
#         grid_yx=(2, 2),
#         target_size=(530, 520),
#         cell_shape=(520, 520),
#         center_range=(0.5, 0.5),
#         fit_mode="cover",
#         p=1.0
#     )
# ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.4, label_fields=[]))


def main():
    import matplotlib.pyplot as plt
    from utills.DataAugmentation import DataAugmentation
    images_dir = "Datatest/img1"
    labels_file = "Datatest/gt/gt.txt"
    dataset = Path22Dataset(images_dir, labels_file)
    mosaic_data = [
        {
            'image': dataset[1][0],
            'bboxes': (dataset[1][1])
        },
        {
            'image': dataset[2][0],
            'bboxes': (dataset[2][1])
        },
        {
            'image': dataset[3][0],
            'bboxes': (dataset[3][1])
        }
    ]
    trans = DataAugmentation(
        image=dataset[1][0],
        bboxes=dataset[1][1],
        mosaic_metadata=mosaic_data
    )
    from utills.drawBoundingBoxes import drawBboxes
    transformed_image = trans['image']
    boxes = trans['bboxes']
    tran = drawBboxes(transformed_image, boxes, class_names=None, colors=None, thickness=2)



if __name__ == "__main__":
    main()
