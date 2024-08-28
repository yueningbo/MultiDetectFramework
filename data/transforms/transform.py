import torch
import torchvision.transforms.functional as TF
from torchvision import transforms


class YOLOv1Transform:
    def __init__(self, size=(416, 416)):
        self.size = size

    def __call__(self, image, annotations):
        # Resize image to the fixed size
        image = TF.resize(image, self.size)

        # Adjust annotations accordingly
        img_width, img_height = image.size
        scale = torch.tensor([img_width, img_height, img_width, img_height], dtype=torch.float32)
        annotations[:, [1, 2, 3, 4]] *= scale

        # Convert image to tensor
        image = TF.to_tensor(image)

        return image, annotations

# Example usage:
# transform = CustomTransform()
# image, annotations = transform(image, annotations)
