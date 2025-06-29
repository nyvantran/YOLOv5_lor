import numpy as np
import albumentations as A
import cv2

# Prepare primary data
primary_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
primary_mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
primary_bboxes = np.array([[10, 10, 40, 40], [50, 50, 90, 90]], dtype=np.float32)
primary_labels = [1, 2]

# Prepare additional images for mosaic
additional_image1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
additional_mask1 = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
additional_bboxes1 = np.array([[20, 20, 60, 60]], dtype=np.float32)
additional_labels1 = [3]

additional_image2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
additional_mask2 = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
additional_bboxes2 = np.array([[30, 30, 70, 70]], dtype=np.float32)
additional_labels2 = [4]

additional_image3 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
additional_mask3 = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
additional_bboxes3 = np.array([[5, 5, 45, 45]], dtype=np.float32)
additional_labels3 = [5]

# Create metadata for additional images - structured as a list of dicts
mosaic_metadata = [
    {
        'image': additional_image1,
        'mask': additional_mask1,
        'bboxes': additional_bboxes1,
        'labels': additional_labels1
    },
    {
        'image': additional_image2,
        'mask': additional_mask2,
        'bboxes': additional_bboxes2,
        'labels': additional_labels2
    },
    {
        'image': additional_image3,
        'mask': additional_mask3,
        'bboxes': additional_bboxes3,
        'labels': additional_labels3
    }
]

# Create the transform with Mosaic
transform = A.Compose([
    A.Mosaic(
        grid_yx=(2, 2),
        target_size=(200, 200),
        cell_shape=(120, 120),
        center_range=(0.4, 0.6),
        fit_mode="cover",
        p=1.0
    ),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

# Apply the transform
transformed = transform(
    image=primary_image,
    mask=primary_mask,
    bboxes=primary_bboxes,
    labels=primary_labels,
    mosaic_metadata=mosaic_metadata  # Pass the metadata using the default key
)

# Access the transformed data
mosaic_image = transformed['image']  # Combined mosaic image
mosaic_mask = transformed['mask']  # Combined mosaic mask
mosaic_bboxes = transformed['bboxes']  # Combined and repositioned bboxes
mosaic_labels = transformed['labels']  # Combined labels from all images
# Display the results
import matplotlib.pyplot as plt
plt.imshow(mosaic_image)
plt.show()
plt.imshow(mosaic_mask)
plt.show()

