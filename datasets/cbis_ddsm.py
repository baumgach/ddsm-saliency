import os
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, RandomSampler, Sampler
from torchvision import transforms

import pydicom
import pandas as pd


class CBISDDSM(Dataset):
    def __init__(
        self,
        root_path,
        split="train",
        image_size=224,
        with_concepts=False,
        normalization=True,
        transform=None,
    ):
        super(CBISDDSM, self).__init__()

        label_file = os.path.join(root_path, f"mass_case_description_{split}_set.csv")
        df = pd.read_csv(label_file)

        img_paths = df["cropped image file path"]
        self.img_paths = [os.path.join(root_path, "CBIS-DDSM", p) for p in img_paths]
        # self.img_paths = [p for p in img_paths if os.path.isfile(p)]
        self.labels = [
            1 if x == "MALIGNANT" else 0
            for i, x in enumerate(df["pathology"])
            # if os.path.isfile(img_paths[i])
        ]
        self.with_concepts = with_concepts
        if self.with_concepts:
            concept_names = ["mass shape", "mass margins"]
            # concepts = [
            #    sorted(set((c for c in df[cn] if not pd.isna(c))))
            #    for cn in concept_names
            # ]
            concepts = [
                [
                    "ARCHITECTURAL_DISTORTION",
                    "ASYMMETRIC_BREAST_TISSUE",
                    "FOCAL_ASYMMETRIC_DENSITY",
                    "IRREGULAR",
                    "IRREGULAR-ARCHITECTURAL_DISTORTION",
                    "IRREGULAR-FOCAL_ASYMMETRIC_DENSITY",
                    "LOBULATED",
                    "LOBULATED-ARCHITECTURAL_DISTORTION",
                    "LOBULATED-IRREGULAR",
                    "LOBULATED-LYMPH_NODE",
                    "LOBULATED-OVAL",
                    "LYMPH_NODE",
                    "OVAL",
                    "OVAL-LYMPH_NODE",
                    "ROUND",
                    "ROUND-IRREGULAR-ARCHITECTURAL_DISTORTION",
                    "ROUND-LOBULATED",
                    "ROUND-OVAL",
                ],
                [
                    "CIRCUMSCRIBED",
                    "CIRCUMSCRIBED-ILL_DEFINED",
                    "CIRCUMSCRIBED-MICROLOBULATED",
                    "CIRCUMSCRIBED-OBSCURED",
                    "ILL_DEFINED",
                    "ILL_DEFINED-SPICULATED",
                    "MICROLOBULATED",
                    "MICROLOBULATED-ILL_DEFINED",
                    "MICROLOBULATED-ILL_DEFINED-SPICULATED",
                    "MICROLOBULATED-SPICULATED",
                    "OBSCURED",
                    "OBSCURED-ILL_DEFINED",
                    "OBSCURED-ILL_DEFINED-SPICULATED",
                    "OBSCURED-SPICULATED",
                    "SPICULATED",
                ],
            ]
            concept_labels = [
                torch.tensor(
                    [[1 if x == c else 0 for c in concepts[i]] for x in df[cn]]
                )
                for i, cn in enumerate(concept_names)
            ]
            self.concepts = torch.cat(concept_labels, dim=1)

        self.transform = (
            transform
            if transform is not None
            else transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                ]
            )
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        dcm = pydicom.read_file(self.img_paths[index])
        image = self.transform(dcm.pixel_array / 65535).float()
        label = self.labels[index]
        if self.with_concepts:
            return image, self.concepts[index], label
        return image, label
