import os

from .oxford_pets import OxfordPets
from .utils import DatasetBase


template = ['a photo of a {}, a type of flower.']

class OxfordFlowers(DatasetBase):

    dataset_dir = 'oxford_flowers'

    def __init__(self, root):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = self.dataset_dir
        self.label_file = "imagelabels.mat"
        self.lab2cname_file = "cat_to_name.json"
        self.split_path = "split_zhou_OxfordFlowers.json"

        self.template = template

        test = OxfordPets.read_split(self.split_path, self.image_dir)
        
        super().__init__(test=test)