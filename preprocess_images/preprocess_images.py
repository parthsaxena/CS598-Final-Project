import numpy as np
import skimage.transform as skTrans
import nibabel as nib
import pandas as pd
import os
import sys
import time


def normalize_img(img_array):
    maxes = np.quantile(img_array, 0.995, axis=(0, 1, 2))
    return img_array / maxes

def create_dataset(meta, meta_all, path_to_datadir):
    files = os.listdir(path_to_datadir)
    start = '_'
    end = '.nii'
    
    for file in files:
        if file != '.DS_Store':
            path = os.path.join(path_to_datadir, file)
            print(path)
            img_id = file.split(start)[-1].split(end)[0]
            idx = meta[meta["Image Data ID"] == img_id].index[0]
            im = nib.load(path).get_fdata()
            n_i, n_j, n_k = im.shape
            center_i = (n_i - 1) // 2  
            center_j = (n_j - 1) // 2
            center_k = (n_k - 1) // 2
            im1 = skTrans.resize(im[center_i, :, :], (72, 72), order=1, preserve_range=True)
            im2 = skTrans.resize(im[:, center_j, :], (72, 72), order=1, preserve_range=True)
            im3 = skTrans.resize(im[:, :, center_k], (72, 72), order=1, preserve_range=True)
            im = np.array([im1, im2, im3]).T
            print(im.shape)
            label = meta.at[idx, "Group"]
            subject = meta.at[idx, "Subject"]
            norm_im = normalize_img(im)
            
            # Creating a temporary DataFrame and concatenating it
            temp_df = pd.DataFrame([{"img_array": norm_im, "label": label, "subject": subject}])
            meta_all = pd.concat([meta_all, temp_df], ignore_index=True)
    
    # Save the final DataFrame
    meta_all.to_pickle("mri_meta.pkl")



def main():
    args = sys.argv[1:]
    path_to_meta = args[0] 
    path_to_datadir = args[1]
    print(path_to_meta)

 
    meta = pd.read_csv(path_to_meta)
    print("opened meta")
    print(len(meta))
    #get rid of not needed columns
    meta = meta[["Image Data ID", "Group", "Subject"]] #MCI = 0, CN =1, AD = 2
    meta["Group"] = pd.factorize(meta["Group"])[0]
    #initialize new dataset where arrays will go
    meta_all = pd.DataFrame(columns = ["img_array","label","subject"])
    create_dataset(meta, meta_all, path_to_datadir)
    
if __name__ == '__main__':
    main()
    
