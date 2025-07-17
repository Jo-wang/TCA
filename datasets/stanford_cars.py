import os

from .oxford_pets import OxfordPets
from .utils import DatasetBase


templates = [
    'a photo of a {}.',
    'a photo of the {}.',
    'a photo of my {}.',
    'i love my {}!',
    'a photo of my dirty {}.',
    'a photo of my clean {}.',
    'a photo of my new {}.',
    'a photo of my old {}.',
]

class StanfordCars(DatasetBase):

    dataset_dir = ''

    def __init__(self, root):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_path = "split_zhou_StanfordCars.json"

        self.template = templates

        test = OxfordPets.read_split(self.split_path, self.dataset_dir)

        super().__init__(test=test)