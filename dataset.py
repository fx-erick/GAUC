import os
from ast import literal_eval
from collections.abc import MutableSequence
from pathlib import Path
from pprint import pprint
from typing import Dict

import pandas as pd
from tabulate import tabulate
import numpy as np


PathLike = str | Path


def load_img_embeddings(file_path):
    feat_array = np.load(file_path, allow_pickle=True)

    if feat_array.dtype.kind != 'V':
        vector_len = feat_array[0][1].shape[0]
        struct = np.dtype([('name', 'U100'), ('vector', 'f4', (vector_len,))])
        feat_array = np.array([(name, vec) for name, vec in feat_array], dtype=struct)

    return feat_array


class CoresetDataset:
    def __init__(
        self,
        datafile_path: PathLike,
        use_only: MutableSequence,
        dataset_vectors_path: PathLike,
        label_replacements: Dict = None,
        use_tiles=None,
        most_similar_last=True,
        take_random=False,
        candidates = None
    ):
        data = pd.read_csv(datafile_path, converters={"path": literal_eval}, usecols=["fname", "label", "path"])
        data = data[data["label"].isin(use_only)]

        if use_tiles:
            data = data[data["fname"].str.contains(use_tiles)]

        self.data = data.sample(frac=1).reset_index(drop=True)  
        self.label_replacements = label_replacements
        self.use_only = use_only
        self.dataset_vectors_path = dataset_vectors_path
        print("Loading dataset vectors...")
        self.dataset_vectors = load_img_embeddings(dataset_vectors_path)
        self.most_similar_last = most_similar_last
        self.take_random = take_random
        self.candidates = candidates

    def __call__(self, num_shots: int = 0, show_bbox: bool = False):
        for i in range(len(self.data)):
            target_image_path = self.data.iloc[i]["path"][0] 
            target_image_fname = self.data.iloc[i]["fname"]
            target_image_label = self.data.iloc[i]["label"]

            if num_shots == 0:
                yield {
                    "image_path": target_image_path,
                    "fname": target_image_fname,
                    "label": target_image_label,
                }
        
            if num_shots > 0:

                

                multi_shot_mappings = {}
                for label, samples in self.candidates.items():
                    multi_shot_mappings[self.label_replacements[label]] = [str(tks) for tks in self.candidates]

                yield {
                    "image_path": target_image_path,
                    "fname": target_image_fname,
                    "label": target_image_label,
                    "multi_shot_mappings": {**multi_shot_mappings},
                }

            


    def pprint_self(self):
        terminal_width = os.get_terminal_size().columns
        pd.set_option("display.width", terminal_width)
        print(tabulate(self.data, headers="keys", tablefmt="psql", showindex=False))
        # pd.reset_option('all')

    def get_multi_shot_mappings(self):
        multi_shot_mappings = {}
        for label, samples in self.candidates.items():
            
            multi_shot_mappings[self.label_replacements[label]] = [str(tks) for tks in samples]
        
        return multi_shot_mappings
