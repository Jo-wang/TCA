import os

from .utils import DatasetBase

from .oxford_pets import OxfordPets


templates = [
    'a photo of a person {}.',
    'a video of a person {}.',
    'a example of a person {}.',
    'a demonstration of a person {}.',
    'a photo of the person {}.',
    'a video of the person {}.',
    'a example of the person {}.',
    'a demonstration of the person {}.',
    'a photo of a person using {}.',
    'a video of a person using {}.',
    'a example of a person using {}.',
    'a demonstration of a person using {}.',
    'a photo of the person using {}.',
    'a video of the person using {}.',
    'a example of the person using {}.',
    'a demonstration of the person using {}.',
    'a photo of a person doing {}.',
    'a video of a person doing {}.',
    'a example of a person doing {}.',
    'a demonstration of a person doing {}.',
    'a photo of the person doing {}.',
    'a video of the person doing {}.',
    'a example of the person doing {}.',
    'a demonstration of the person doing {}.',
    'a photo of a person during {}.',
    'a video of a person during {}.',
    'a example of a person during {}.',
    'a demonstration of a person during {}.',
    'a photo of the person during {}.',
    'a video of the person during {}.',
    'a example of the person during {}.',
    'a demonstration of the person during {}.',
    'a photo of a person performing {}.',
    'a video of a person performing {}.',
    'a example of a person performing {}.',
    'a demonstration of a person performing {}.',
    'a photo of the person performing {}.',
    'a video of the person performing {}.',
    'a example of the person performing {}.',
    'a demonstration of the person performing {}.',
    'a photo of a person practicing {}.',
    'a video of a person practicing {}.',
    'a example of a person practicing {}.',
    'a demonstration of a person practicing {}.',
    'a photo of the person practicing {}.',
    'a video of the person practicing {}.',
    'a example of the person practicing {}.',
    'a demonstration of the person practicing {}.',
]

class UCF101(DatasetBase):

    dataset_dir = 'ucf101'

    def __init__(self, root):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, '')
        self.split_path = "split_zhou_UCF101.json"

        self.template = templates

        test = OxfordPets.read_split(self.split_path, self.image_dir)
    
        super().__init__(test=test)
