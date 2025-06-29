import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import supervision as sv
import random


def drawBoundingBoxes(image, bboxes, class_names=None, colors=None, line_width=2):
    """
    Vẽ bounding boxes lên hình ảnh PIL

    Args:
        image: PIL Image object
        bboxes: List các bounding box với format [class_id, x_center, y_center, width, height]
                Trong đó x_center, y_center, width, height đều trong khoảng [0-1]
        class_names: Dict hoặc list tên các class (optional)
        colors: List màu sắc cho từng class (optional)
        line_width: Độ dày của đường viền

    Returns:
        PIL Image với bounding boxes đã vẽ
    """
    # Tạo bản copy để không thay đổi ảnh gốc
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    # Lấy kích thước ảnh
    img_width, img_height = img_copy.size
    # img_width, img_height = 1, 1

    # Tạo màu sắc ngẫu nhiên nếu không được cung cấp
    if colors is None:
        colors = []
        for i in range(100):  # Tạo 100 màu khác nhau
            colors.append((
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            ))

    # Thử load font, nếu không có thì dùng font mặc định
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    for bbox in bboxes:
        class_id, x_center, y_center, width, height = bbox

        # Chuyển đổi từ tọa độ normalized sang tọa độ pixel
        x_center_px = x_center * img_width
        y_center_px = y_center * img_height
        width_px = width * img_width
        height_px = height * img_height

        # Tính tọa độ góc trên trái và góc dưới phải
        x1 = x_center_px - width_px / 2
        y1 = y_center_px - height_px / 2
        x2 = x_center_px + width_px / 2
        y2 = y_center_px + height_px / 2

        # Đảm bảo tọa độ nằm trong giới hạn ảnh

        # Chọn màu cho class
        color = colors[int(class_id) % len(colors)]

        # Vẽ bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # Tạo label text
        if class_names is not None:
            if isinstance(class_names, dict):
                label = class_names.get(int(class_id), f"Class {int(class_id)}")
            elif isinstance(class_names, list):
                label = class_names[int(class_id)] if int(class_id) < len(class_names) else f"Class {int(class_id)}"
            else:
                label = f"Class {int(class_id)}"
        else:
            label = f"Class {int(class_id)}"

        # Vẽ background cho text
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Vẽ background cho text
        draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1],
                       fill=color, outline=color)

        # Vẽ text
        draw.text((x1 + 2, y1 - text_height - 2), label, fill='white', font=font)

    return img_copy


# Ví dụ sử dụng
def example_usage():
    """
    Ví dụ cách sử dụng hàm draw_bounding_boxes
    """
    # Tạo ảnh ví dụ (thay thế bằng ảnh thật của bạn)
    # image = Image.open("path/to/your/image.jpg")

    # Hoặc tạo ảnh test
    image = Image.new('RGB', (640, 480), color='lightblue')

    # Định nghĩa bounding boxes
    # Format: [class_id, x_center, y_center, width, height]
    bboxes = [
        [0, 0.3, 0.4, 0.2, 0.3],  # Class 0, tâm tại (0.3, 0.4), kích thước 0.2x0.3
        [1, 0.7, 0.6, 0.15, 0.25],  # Class 1, tâm tại (0.7, 0.6), kích thước 0.15x0.25
        [2, 0.5, 0.2, 0.1, 0.15],  # Class 2, tâm tại (0.5, 0.2), kích thước 0.1x0.15
    ]

    # Định nghĩa tên các class (optional)
    class_names = {
        0: "Person",
        1: "Car",
        2: "Bicycle"
    }

    # Vẽ bounding boxes
    result_image = drawBoundingBoxes(image, bboxes, class_names=class_names)

    # Hiển thị hoặc lưu ảnh
    result_image.show()
    # result_image.save("output_with_bboxes.jpg")

    return result_image


def drawBboxes(image, bboxes, class_names=None, colors=None, thickness=2):
    """
    Vẽ bounding boxes theo format YOLO lên ảnh

    Args:
        image: Ảnh được đọc bằng cv2.imread()
        bboxes: List các bounding box theo format YOLO
                Mỗi bbox có thể là:
                - [class_id, x_center, y_center, width, height] (5 elements)
                - [x_center, y_center, width, height] (4 elements)
                Các giá trị tọa độ được normalize (0-1)
        class_names: List tên các class (optional)
        colors: List màu sắc cho từng class (optional)
        thickness: Độ dày của đường viền bbox

    Returns:
        image: Ảnh đã được vẽ bounding boxes
    """

    # Lấy kích thước ảnh
    h, w = image.shape[:2]

    # Tạo bản copy để không thay đổi ảnh gốc
    img_copy = image.copy()

    # Màu mặc định nếu không được cung cấp
    if colors is None:
        colors = [
            (0, 255, 0),  # Xanh lá
            (255, 0, 0),  # Đỏ
            (0, 0, 255),  # Xanh dương
            (255, 255, 0),  # Vàng
            (255, 0, 255),  # Tím
            (0, 255, 255),  # Cyan
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
        ]

    for i, bbox in enumerate(bboxes):
        # Xử lý format khác nhau của bbox
        if len(bbox) == 5:
            x_center, y_center, width, height, class_id = bbox
            class_id = int(class_id)
        elif len(bbox) == 4:
            x_center, y_center, width, height = bbox
            class_id = 0
        else:
            print(f"Warning: Bbox {i} có format không hợp lệ: {bbox}")
            continue

        # Chuyển đổi từ format YOLO sang tọa độ pixel
        x_center_pixel = int(x_center * w)
        y_center_pixel = int(y_center * h)
        width_pixel = int(width * w)
        height_pixel = int(height * h)

        # Tính tọa độ góc trên trái và góc dưới phải
        x1 = int(x_center_pixel - width_pixel / 2)
        y1 = int(y_center_pixel - height_pixel / 2)
        x2 = int(x_center_pixel + width_pixel / 2)
        y2 = int(y_center_pixel + height_pixel / 2)

        # Đảm bảo tọa độ nằm trong phạm vi ảnh
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        # Chọn màu cho bbox
        color = colors[class_id % len(colors)]

        # Vẽ bounding box
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)

        # Vẽ label nếu có class_names
        if class_names and class_id < len(class_names):
            label = class_names[class_id]

            # Tính kích thước text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            text_thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, text_thickness
            )

            # Vẽ background cho text
            cv2.rectangle(
                img_copy,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )

            # Vẽ text
            cv2.putText(
                img_copy,
                label,
                (x1, y1 - baseline - 2),
                font,
                font_scale,
                (255, 255, 255),
                text_thickness
            )

    return img_copy


# Chạy ví dụ
if __name__ == "__main__":
    example_usage()
