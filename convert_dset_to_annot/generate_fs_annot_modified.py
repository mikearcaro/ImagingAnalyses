import nibabel as nib
import os 
import numpy as np
import pandas as pd

# Iterate over all 4-letter long folders in the current directory
for curr_folder in [d for d in os.listdir() if os.path.isdir(d) and len(d) == 4]:
# %%

    
    # %% load roi values
    f_rh_path = os.path.join(curr_folder, 'rois', 'allparietal-rh.1D.dset')  ##### change path ######
    data_rh = pd.read_csv(f_rh_path, delim_whitespace=True, comment='#', header=None)
    data_rh = data_rh.rename(columns={0:'node', 1:'roi'})
    
    # load surf geometry file to get vert size
    g_rh_path = os.path.join(curr_folder, 'surf', 'rh.inflated')  ##### change path ######
    surf_size = nib.freesurfer.io.read_geometry(g_rh_path)[0].shape[0]
    
    # generate roi values for fullsize surf
    data_rh_all = pd.DataFrame(np.arange(0,surf_size-1)[...,None], columns=['node']) 
    data_rh_all = pd.merge(data_rh_all, data_rh, on='node', how='left')
    data_rh_all = data_rh_all.fillna(0)  # fill vert w/o roi label with 0
    data_rh_all = data_rh_all.astype({'roi':int})
    
    # color table
    ctab_data = (np.random.rand(7,4) * 255).astype(int)
    ctab_data = np.concatenate([(np.ones([1,4])*255).astype(int), ctab_data])
    ctab_data = np.concatenate([ctab_data, np.arange(8).astype(int)[...,None]], axis=1)
    
    # save new annot file
    nib.freesurfer.io.write_annot(os.path.join(curr_folder, 'rois', 'allparietal-rh.annot'), 
                                  data_rh_all['roi'].values, ctab_data, 
                                  list(np.arange(8).astype(str)))
    