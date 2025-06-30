import albumentations as A

# configurations for data augmentation
output_size = (640, 640)
center_range = (0.5, 0.5)
fit_mode = "cover"
p = 0.5  # Probability of applying the augmentation
scale_limit = (-0.1, 0.1)  # Scale range for RandomAffine
translate_percent = (-0.15, 0.15)  # Translation percentage for RandomAffine

# Define the data augmentation pipeline using Albumentations

# Mosaic augmentation for object detection tasks
MosaicAugmentation = A.Mosaic(
    grid_yx=(2, 2),
    # target_size=output_size,
    # cell_shape=output_size,
    target_size=(640, 640),
    cell_shape=(640, 640),
    center_range=center_range,
    # fit_mode="cover",
    p=p
)

RandomAffine = A.Compose([
    A.SafeRotate(
        limit=(-90, 90),
        rotate_method="ellipse",
        fill=0,
        fill_mask=0,
        p=p
    ),
    A.RandomScale(
        scale_limit=scale_limit,
        p=p
    ),
    A.Affine(
        translate_percent=(-0.1, 0.1),
        p=p
    ),
    A.RandomCrop(
        height=output_size[0],
        width=output_size[1],
        pad_if_needed=True,
        p=p
    )
], p=p,
)

RandomHorizontalFlip = A.HorizontalFlip(p=p)

DataAugmentation = A.Compose([
    MosaicAugmentation,
    RandomAffine,
    RandomHorizontalFlip,
    A.Resize(height=output_size[0], width=output_size[1], p=1.0)
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.4, label_fields=[]), p=p)
