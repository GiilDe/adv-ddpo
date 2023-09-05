import torch
import torchvision
from PIL import Image


def normalize_to_unit_interval(images):
    """Normalizes the output of a network (persumeably in the range [-1,1]) to values in [0,1]

    Args:
        images torch.tensor: batch of images with mean 0 and std of 1

    Returns:
        torch.tesor: batch of normalized images
    """    
    return ((images / 2) + 0.5).clamp(0,1)


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def normalized_tensor_to_npimg(tensor):
    image = normalize_to_unit_interval(tensor)
    if tensor.ndim == 4:
        image = image.cpu().permute(0, 2, 3, 1).numpy()
    else:
        image = image.cpu().permute(1, 2, 0).numpy()
    return image


def tensor_to_pil(tensor):
    normalized = normalize_to_unit_interval(tensor)
    return normalized_tensor_to_pil(normalized)


def normalized_tensor_to_pil(tensor):
    np_img = normalized_tensor_to_npimg(tensor)
    return numpy_to_pil(np_img)

def create_comparison_img(left_img, right_img):
    img_comparisons = torch.stack((left_img, right_img), dim=1).view(left_img.shape[0] * 2, left_img.shape[1], left_img.shape[2], left_img.shape[3])
    img_comparisons = [numpy_to_pil(torchvision.utils.make_grid(img_comparisons[j: j+2]).cpu().permute(1, 2, 0).numpy())[0] for j in range(0,left_img.shape[0] * 2,2)]
    return img_comparisons