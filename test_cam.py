import pydicom
import matplotlib.pyplot as plt
import torch

#%%
import get_cam
cam_model, dataset = get_cam.main()

#%%
import pandas as pd
import os

root_path = "/home/stefano/Datasets/CBIS-DDSM/"
label_file = os.path.join(root_path, "mass_case_description_train_set.csv")
df = pd.read_csv(label_file)

labels = df["pathology"]

cropped_img_paths = df["cropped image file path"]
cropped_img_paths = [
    os.path.join(root_path, "CBIS-DDSM", p)
    for p in cropped_img_paths
]

mask_paths = df["ROI mask file path"]
mask_paths = [
    os.path.join(
        root_path, "CBIS-DDSM", p.replace("\n", "")
    )
    for p in mask_paths
]

uncropped_img_paths = df["image file path"]
uncropped_img_paths = [
    os.path.join(root_path, "CBIS-DDSM", p)
    for p in uncropped_img_paths
]

inds = df.index[df["patient_id"] == "P_00092"]

#%%

concept_names = ['mass shape', 'mass margins']
concepts = [sorted(set((c for c in df[cn] if not pd.isna(c)))) for cn in concept_names]
concepts_flat = concepts[0] + concepts[1]
concept_labels = [torch.tensor([[1 if x == c else 0 for c in concepts[i]] for x in df[cn]]) for i, cn in enumerate(concept_names)]
con = torch.cat(concept_labels, dim=1)

#%%
from PIL import Image
from torchvision import transforms
topil = transforms.ToPILImage()

for i in inds:
    im, co, la = dataset[i]
    img = topil(im)
    saveto = f"{df['patient_id'][i]}_{df['left or right breast'][i]}_{df['image view'][i]}_abnormality{df['abnormality id'][i]}_{labels[i]}.png"
    img.save(os.path.join("cams", saveto), "PNG")
    for j, conlab in enumerate(concepts_flat):
        single_label = torch.eye(co.size(0), device=co.device)[None, j]
        single_cam = cam_model(im[None,:,:,:], single_label, back=0, grad_outputs=single_label)
        img = topil(single_cam[0])
        saveto = f"{df['patient_id'][i]}_{df['left or right breast'][i]}_{df['image view'][i]}_abnormality{df['abnormality id'][i]}_{labels[i]}_{conlab}({'TRUE' if co[j] else 'FALSE'}).png"
        img.save(os.path.join("cams", saveto), "PNG")
