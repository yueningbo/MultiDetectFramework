import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch


def visualize_predictions(images, predictions):
    """
    Visualize the predictions on the images.
    """
    for i, (image, prediction) in enumerate(zip(images, predictions)):
        image = image.permute(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
        image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        image = image.numpy()
        plt.imshow(image)
        ax = plt.gca()

        for box in prediction[:, :4]:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        plt.show()