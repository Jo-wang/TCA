import os

from .utils import DatasetBase
from .oxford_pets import OxfordPets


templates = [
    'a photo of a {}.',
    'a painting of a {}.',
    'a plastic {}.',
    'a sculpture of a {}.',
    'a sketch of a {}.',
    'a tattoo of a {}.',
    'a toy {}.',
    'a rendition of a {}.',
    'a embroidered {}.',
    'a cartoon {}.',
    'a {} in a video game.',
    'a plushie {}.',
    'a origami {}.',
    'art of a {}.',
    'graffiti of a {}.',
    'a drawing of a {}.',
    'a doodle of a {}.',
    'a photo of the {}.',
    'a painting of the {}.',
    'the plastic {}.',
    'a sculpture of the {}.',
    'a sketch of the {}.',
    'a tattoo of the {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'the embroidered {}.',
    'the cartoon {}.',
    'the {} in a video game.',
    'the plushie {}.',
    'the origami {}.',
    'art of the {}.',
    'graffiti of the {}.',
    'a drawing of the {}.',
    'a doodle of the {}.',
]


class Caltech101(DatasetBase):

    

    def __init__(self, root):
        self.dataset_dir = 'caltech101'
        
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, '101_ObjectCategories')
        self.split_path = "split_zhou_Caltech101.json"

        self.template = templates

        test = OxfordPets.read_split(self.split_path, self.image_dir)

        super().__init__(test=test)