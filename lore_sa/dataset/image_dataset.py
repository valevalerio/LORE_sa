from lore_sa.dataset import Dataset


class ImageDataset():
    """Image dataset."""

    def __init__(self, descriptor: dict, transform=None):
        """
        Args:
            descriptor (dict): it contains the essential information regarding the image dataset. Format:
                >>> {
                'shape': [xxx, yyy],
                'channels': <number of channels>,
                'class_names': <list of class names>,
                'input_dim': <input dimension>, # e.g. np.prod(shape)
                }
            transform (callable, optional): Optional transform to be applied
        """
        self.descriptor = descriptor
        self.transform = transform

    @classmethod
