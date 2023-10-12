from torchvision.datasets import CIFAR10
from typing import Any, Callable, Optional, Tuple
from PIL import Image
import numpy as np

class CIFAR10Pair(CIFAR10):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            fixed_noise: bool =False
    ) -> None:

        super(CIFAR10Pair, self).__init__(root, train=train, transform=transform,
                                      target_transform=target_transform, download=download)
        self.fixed_noise = fixed_noise
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        shape = img.shape
        if self.fixed_noise:
            np.random.seed(index)
            
        rimg = np.random.randn( shape[2], shape[0], shape[1])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, rimg
