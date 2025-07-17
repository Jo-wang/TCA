import os

from .utils import DatasetBase
from .oxford_pets import OxfordPets


templates = [
    'a photo of a {} texture.',
    'a photo of a {} pattern.',
    'a photo of a {} thing.',
    'a photo of a {} object.',
    'a photo of the {} texture.',
    'a photo of the {} pattern.',
    'a photo of the {} thing.',
    'a photo of the {} object.',
]

class DescribableTextures(DatasetBase):

    dataset_dir = 'dtd'

    def __init__(self, root):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        self.split_path = "split_zhou_DescribableTextures.json"

        self.template = templates

        test = OxfordPets.read_split(self.split_path, self.image_dir)

        super().__init__(test=test)
