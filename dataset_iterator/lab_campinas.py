import os
from dataset_iterator import SRDatasetIterator

class LabCampinasSift(SRDatasetIterator):

    def __init__(self, dataset_root):
        SRDatasetIterator.__init__(self, dataset_root,
                                   HR_subfolder="aligned/sift")

class LabCampinasEcc(SRDatasetIterator):

    def __init__(self, dataset_root):
        SRDatasetIterator.__init__(self, dataset_root,
                                   HR_subfolder="aligned/ecc")
