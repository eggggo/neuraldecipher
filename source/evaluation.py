#!/usr/bin/env python
# coding: utf-8

# ## Evaluation script
# This notebook contains a minimal example for the reconstruction of ECFP$_6$ representations of length 1024 on the cluster split.  
# Make sure to execute the `source/run_cddd_inference_server.py` in another process within the `neuraldecipher` environment, such that the CDDD inference server can be used to decode the predicted cddd-representations back to SMILES representations.  
# To run the CDDD-server on three GPUs 0,1,2 with 6 parallel processes, execute:
# ````
# python source/run_cddd_inference_server.py --device 0,1,2 --nservers 6
# 
# ````
# On the console following message should be printed out:
# ```
# Using GPU devices: 0,1,2
# Total number of servers to spin up: 6
# Server running on GPU  0
# Server running on GPU  0
# Server running on GPU  1
# Server running on GPU  1
# Server running on GPU  2
# Server running on GPU  2
# ```
# You can additionally check if the GPU-0 device is blocked by simply executing:
# `nvidia-smi`  

# #### Load needed cddd modules

# In[1]:


from cddd.inference import InferenceServer


# #### Instantiate the CDDD-Inference server

# In[2]:


inference_server = InferenceServer(port_frontend=5527, use_running=True)


# #### Load rest modules for reverse-engineering

# In[3]:


import torch
import numpy as np
import os
import h5py
from torch.utils.data import Dataset, DataLoader


# In[4]:


from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
import rdkit.Chem as Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem


# In[5]:


import os
import json
from multiprocessing import Pool


# #### Load utility modules for dataloading and the Neuraldecipher class

# In[6]:


from utils import create_train_and_test_set, create_data_loaders, get_eval_data
from models import Neuraldecipher


# In[7]:


def get_data_loaders(ecfp_path, random_split=False):
    train_data, test_data = create_train_and_test_set(ecfp_path, random_split=random_split)
    train_loader, test_loader = create_data_loaders(train_data, test_data, batch_size=256, num_workers=5, shuffle_train=False)
    return train_loader, test_loader


# In[8]:


def load_model(param_config_path, model_weights_path):
    """
    Loads the neuraldecipher model
    :param param_config_path [str] path to where the parameter configurations are stored
    :param model_weights_path [str] path to where the model weights are stored
    """
    with open(param_config_path, 'r', encoding='utf-8') as config_file:
        json_string = config_file.read()
    
    print("Parameter configs:")
    print(json_string)
    print("-"*100)
    print("Model:")
    params = json.loads(json_string)
    nd_model = Neuraldecipher(**params['neuraldecipher'])
    nd_model.load_state_dict(torch.load(model_weights_path, map_location='cuda:0'))
    print(nd_model)
    return nd_model


# In[9]:


neuraldecipher = load_model("../params/1024_config_count_gpu.json",
                           "../models/1024_final_model_cs_gpu/weights.pt")


# ### Set device for current `neuraldecipher`.
# If the Neuraldecipher fits into GPU memory with GPU:0 next to the CDDD inference server, you can allocate the model there.  
# However, we recommend using another GPU.
# 

# In[10]:


device = torch.device("cuda:0")
neuraldecipher = neuraldecipher.eval()    
neuraldecipher = neuraldecipher.to(device)


# In[11]:


def forward_pass_dataloader(dataloader, device, neuraldecipher, true_smiles=None):
    """
    Computes a full forward pass on an entire dataset
    :param dataloader [torch.utils.data.Dataloader] Torch dataloader that contains the batches
    :param device [torch.device] Torch device where the computation should be performed on
    :param neuraldecipher [Neuraldecipher] neuraldecipher model
    :param true_smiles [None or list] List of true smiles representation. This variable is used when the dataloader
                        does not contain the true SMILES representations within each batch.
                        (The case when dealing with temporal split)
    """
    predicted_cddd = []
    with torch.no_grad():
        if true_smiles is None:
            true_smiles = []
            for sample_batched in dataloader:
                ecpf_in = sample_batched['ecfp'].to(device=device, dtype=torch.float32)
                true_smiles.append(sample_batched['smiles'])
                output = neuraldecipher(ecpf_in) 
                predicted_cddd.append(output.detach().cpu().numpy())
            
            true_smiles = np.concatenate(true_smiles)
        else:
            for batch in dataloader:
                ecpf_in = batch.to(device=device, dtype=torch.float32)
                output = neuraldecipher(ecpf_in) 
                predicted_cddd.append(output.detach().cpu().numpy())

    predicted_cddd = np.concatenate(predicted_cddd)

    return predicted_cddd, true_smiles


# In[12]:


### Utility function to create batches from a large list
def get_batches_from_large_list(large_list, batch_size):
    n_batches = len(large_list) // batch_size
    rest_indices = len(large_list) - n_batches*batch_size
    last_start = n_batches*batch_size
    last_end = last_start + rest_indices
    batches = [large_list[i*batch_size:(i+1)*batch_size] for i in range(n_batches)]
    batches.append(large_list[last_start:last_end])
    return batches


# In[13]:


def canonicalize_sanitize_smiles(smiles, sanitize=True):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if sanitize:
            Chem.SanitizeMol(mol)
        smi = Chem.MolToSmiles(mol)
        return smi
    except:
        None


# In[14]:


def get_similarity(smi_true,
                   smi_recon,
                   radius=3, nbits=1024):
    """
    For evaluation always compute the ECFP fingerprints on consistens lengths and radius.
    Evaluation settings are ECFP6_1024
    """
    mol_true = Chem.MolFromSmiles(smi_true)
    mol_reconstructed = Chem.MolFromSmiles(smi_recon)
    #fingerprint similarity according to ECFP_fixed
    fp1_ecfp = AllChem.GetHashedMorganFingerprint(mol_true, radius=radius, nBits=nbits)
    fp2_ecfp = AllChem.GetHashedMorganFingerprint(mol_reconstructed, radius=radius, nBits=nbits)
    tanimoto_ecfp = DataStructs.TanimotoSimilarity(fp1_ecfp, fp2_ecfp)

    return tanimoto_ecfp


# ### Note:
# Since we are using 6 CDDD inference servers, we can set the pool of workers to 6

# In[15]:


def decode_cddd(batch_list, npool=6):
    with Pool(npool) as pool:
        decoded_smiles = pool.map(inference_server.emb_to_seq, batch_list)
    return decoded_smiles


# #### Get the validation data set

# In[16]:


ecfp_path_validationset = "data/dfFold1024/ecfp6_train_c.npy"


# In[17]:

_, validation_loader = get_data_loaders(ecfp_path_validationset, random_split=False)

# #### Get the temporal data set

# In[18]:


temporal_dataloader, temporal_smiles = get_eval_data(ecfp_path='data/dfFold1024/ecfp6_temporal_c.npy',
                                                     smiles_path='data/smiles_temporal.npy')


# ### Wrapper for Evaluation

# In[19]:


def eval_wrapper(neuraldecipher, dataloader, true_smiles):
    """
    
    """
    # compute full forwardpass of dataloader
    print("Predicting cddd representations...")
    predicted_cddd, true_smiles = forward_pass_dataloader(dataloader, device, neuraldecipher, true_smiles)
    # retrieve string representations with CDDD-decoder network
    predicted_cddd = get_batches_from_large_list(predicted_cddd, 1024)
    print("Decoding predicted cddd representations...")
    decoded_smiles = decode_cddd(predicted_cddd)
    decoded_smiles = np.concatenate(decoded_smiles)
    # canonicalize if possible, returns canonical smiles or None.
    canonical_smiles = [canonicalize_sanitize_smiles(s, sanitize=True) for s in decoded_smiles]
    # check valid SMILES
    valid_ids = [i for i, smi in enumerate(canonical_smiles) if smi!= None]
    validity = len(valid_ids)/len(canonical_smiles)
    print(f"Dataset size: {len(decoded_smiles)}.")
    print(f"Validity of the reconstruction: {np.round(validity, 4)}.")
    valid_recon_smiles = decoded_smiles[valid_ids]
    valid_true_smiles = true_smiles[valid_ids]
    # check reconstruction accuracy
    reconstruction_acc = np.sum([a==b for a,b in zip(valid_recon_smiles, valid_true_smiles)])/len(valid_ids)
    print(f"Reconstruction accuracy: {np.round(reconstruction_acc, 4)}.")
    # get Tanimoto similarity 
    tanimoto_sim = [get_similarity(smi_true, smi_recon) for smi_true, smi_recon in zip(valid_true_smiles,
                                                                                   valid_recon_smiles)]
    print(f"Tanimoto similarity: {np.round(np.mean(tanimoto_sim), 4)}.")
    
    
    res_dict = dict()
    res_dict["true_smiles"] = valid_true_smiles
    res_dict["recon_smiles"] = valid_recon_smiles
    res_dict["tanimoto_sim"] = tanimoto_sim
    
    return {"validity": validity, "recon_acc": reconstruction_acc, "res_dict": res_dict}


# ## Evaluation: Validation dataset from the cluster split (112K samples)

# In[20]:


get_ipython().run_cell_magic('time', '', 'res_validation_nd1024_count = eval_wrapper(neuraldecipher, validation_loader, None)')


# ## Evaluation: Temporal dataset from ChEMBL26 (55K samples)

# In[21]:


get_ipython().run_cell_magic('time', '', 'res_temporal_nd1024_count = eval_wrapper(neuraldecipher, temporal_dataloader, temporal_smiles)')


# #### Finish

# #### Stop the CDDD inference server execution in your other shell 

# In[ ]:




