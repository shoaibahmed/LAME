import numpy as np
from PIL import Image
from .corruptions import *
from torchvision import transforms

corruption_tuple = (gaussian_noise, shot_noise, impulse_noise, defocus_blur,
                    glass_blur, motion_blur, zoom_blur, snow, frost, fog,
                    brightness, contrast, elastic_transform, pixelate, jpeg_compression,
                    speckle_noise, gaussian_blur, spatter, saturate)

corruption_dict = {corr_func.__name__: corr_func for corr_func in corruption_tuple}


def corrupt(x, severity=1, corruption_name=None, corruption_number=-1):
    """
    :param x: image to corrupt; a 224x224x3 numpy array in [0, 255]
    :param severity: strength with which to corrupt x; an integer in [0, 5]
    :param corruption_name: specifies which corruption function to call;
    must be one of 'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
                    'speckle_noise', 'gaussian_blur', 'spatter', 'saturate';
                    the last four are validation functions
    :param corruption_number: the position of the corruption_name in the above list;
    an integer in [0, 18]; useful for easy looping; 15, 16, 17, 18 are validation corruption numbers
    :return: the image x corrupted by a corruption function at the given severity; same shape as input
    """

    if corruption_name:
        x_corrupted = corruption_dict[corruption_name](Image.fromarray(x), severity)
    elif corruption_number != -1:
        x_corrupted = corruption_tuple[corruption_number](Image.fromarray(x), severity)
    else:
        raise ValueError("Either corruption_name or corruption_number must be passed")

    return np.uint8(x_corrupted)


class ImageAugmentator:
    def __init__(self) -> None:
        transforms_list = [
            corruptions.ToPILImage(),
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.75, 1.33), interpolation=Image.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.RandomChoice(
                    [
                        corruptions.GaussianNoise(),
                        corruptions.ShotNoise(),
                        corruptions.ImpulseNoise(),
                        corruptions.SpeckleNoise(),
                        
                        corruptions.GaussianBlur(),
                        corruptions.GlassBlur(),
                        corruptions.DefocusBlur(),
                        corruptions.MotionBlur(),
                        corruptions.ZoomBlur(),
                        
                        corruptions.Fog(),
                        corruptions.Frost(),
                        corruptions.Snow(),
                        corruptions.Spatter(),
                        corruptions.ElasticTransform(),
                        
                        corruptions.Contrast(),
                        corruptions.Brightness(),
                        corruptions.Saturate(),
                        corruptions.JPEGCompression(),
                        corruptions.Pixelate()
                    ]
                )],
                p=0.9,
            ),
            corruptions.ToNumpyArray(),
        ]

        print("Transform list:", transforms_list)
        self.aug_func = transforms.Compose(transforms_list)
    
    def apply_aug(self, x):
        return self.aug_func(x)
