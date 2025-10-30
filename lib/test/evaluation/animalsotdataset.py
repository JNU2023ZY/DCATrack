import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


class AnimalSOTDataset(BaseDataset):

    def __init__(self):
        super().__init__()

        self.base_path = self.env_settings.animalsot_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)
        f1 = open(anno_path, 'r')
        groundtruth = f1.readlines()
        f1.close()
        if groundtruth[0].find(',') > 0:
            ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float)
        else:
            ground_truth_rect = load_text(str(anno_path), delimiter='\t', dtype=np.float)
        frames_path = '{}/{}/img'.format(self.base_path, sequence_name)
        frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")]
        frame_list.sort(key=lambda f: int(f[:-4]))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]

        return Sequence(sequence_name, frames_list, 'animalsot', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        with open('{}/list.txt'.format(self.base_path)) as f:
            sequence_list = f.read().splitlines()
        return sequence_list
