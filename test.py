import pydicom
import matplotlib.pyplot as plt
import torch

#%%
path = "/home/stefano/Datasets/CBIS-DDSM/CBIS-DDSM/Mass-Training_P_00001_LEFT_CC_1/1.3.6.1.4.1.9590.100.1.2.108268213011361124203859148071588939106/1.3.6.1.4.1.9590.100.1.2.296736403313792599626368780122205399650/1-2.dcm"
path = "/home/stefano/Datasets/CBIS-DDSM/CBIS-DDSM/Mass-Training_P_00001_LEFT_CC_1/1.3.6.1.4.1.9590.100.1.2.108268213011361124203859148071588939106/1.3.6.1.4.1.9590.100.1.2.296736403313792599626368780122205399650/1-1.dcm"
path = "/home/stefano/Datasets/CBIS-DDSM/CBIS-DDSM/Mass-Training_P_00001_LEFT_CC/1.3.6.1.4.1.9590.100.1.2.422112722213189649807611434612228974994/1.3.6.1.4.1.9590.100.1.2.342386194811267636608694132590482924515/1-1.dcm"
ds = pydicom.read_file(path)

#%%
plt.figure()
plt.imshow(ds.pixel_array)
plt.show()

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
concept_labels = [torch.tensor([[1 if x == c else 0 for c in concepts[i]] for x in df[cn]]) for i, cn in enumerate(concept_names)]
con = torch.cat(concept_labels, dim=1)

#%%
from PIL import Image

for i in inds:
    for l, pref in zip(
        [uncropped_img_paths, mask_paths, cropped_img_paths], ["uncropped", "mask", "cropped"]
    ):
        path = l[i]
        dcm = pydicom.read_file(path)
        im = Image.fromarray(dcm.pixel_array)
        saveto = f"{df['patient_id'][i]}_{df['left or right breast'][i]}_{df['image view'][i]}_abnormality{df['abnormality id'][i]}_{labels[i]}_{pref}.png"
        im.save(saveto, "PNG")
