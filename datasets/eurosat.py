import os

from .utils import DatasetBase
from .oxford_pets import OxfordPets


templates = [
    'a centered satellite photo of {}.',
    'a centered satellite photo of a {}.',
    'a centered satellite photo of the {}.',
]

NEW_CLASSNAMES = {
    'AnnualCrop': 'Annual Crop Land',
    'Forest': 'Forest',
    'HerbaceousVegetation': 'Herbaceous Vegetation Land',
    'Highway': 'Highway or Road',
    'Industrial': 'Industrial Buildings',
    'Pasture': 'Pasture Land',
    'PermanentCrop': 'Permanent Crop Land',
    'Residential': 'Residential Buildings',
    'River': 'River',
    'SeaLake': 'Sea or Lake'
}


class EuroSAT(DatasetBase):

    dataset_dir = 'eurosat'

    def __init__(self, root):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = self.dataset_dir
        self.split_path = "split_zhou_EuroSat.json"
        
        self.template = templates

        test = OxfordPets.read_split(self.split_path, self.image_dir)
        
        super().__init__(test=test)
