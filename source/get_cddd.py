#!/usr/bin/env python
# coding: utf-8

# ## Get CDDD notebook script
# This notebook contains the script to encode the SMILES representation as CDDD representations
# Make sure to execute the `source/run_cddd_inference_server.py` in another process within the `neuraldecipher` environment, such that the CDDD inference server can be used to encode the SMILES representations to CDDD.  
# To run the CDDD-server on one GPU 0 with 2 parallel processes, execute:
# ````
# python source/run_cddd_inference_server.py --device 0 --nservers 2
# 
# ````
# On the console following message should be printed out:
# ```
# Using GPU devices: 0
# Total number of servers to spin up: 2
# Server running on GPU  0
# Server running on GPU  0
# ```
# You can additionally check if the GPU-0 device is blocked by simply executing:
# `nvidia-smi`

# #### Load needed cddd modules

# In[1]:

from cddd.inference import InferenceServer


# #### Here we will spin 6 CDDD servers distributed on 3 GPUs
# Execute following command in another shell with the `neuraldecipher` environment
# ```
# python source/run_cddd_inference_server.py --device 0,1,2 --nservers 6
# ```

# In[2]:


import numpy as np
from multiprocessing import Pool


# In[3]:


smiles_list = np.load("../data/smiles.npy", allow_pickle=True).tolist()


# In[4]:


print(len(smiles_list))


# In[5]:


inference_server = InferenceServer(port_frontend=5527, use_running=True)


# In[6]:


### Utility function to create batches from a large list
def get_batches_from_large_list(large_list, batch_size):
    n_batches = len(large_list) // batch_size
    rest_indices = len(large_list) - n_batches*batch_size
    last_start = n_batches*batch_size
    last_end = last_start + rest_indices
    batches = [large_list[i*batch_size:(i+1)*batch_size] for i in range(n_batches)]
    batches.append(large_list[last_start:last_end])
    return batches


# In[7]:


smiles_list = get_batches_from_large_list(smiles_list, 1024)


# In[8]:


print(len(smiles_list))


# ### Encode the SMILES representations into CDDDs
# ##### Note:
# Since we are using 6 CDDD inference servers, we can set the pool of workers to 6

# In[9]:


def encode_smiles(batch_list, npool=6):
    with Pool(npool) as pool:
        encoded_cddd = pool.map(inference_server.seq_to_emb, batch_list)
    return encoded_cddd


# In[10]:


get_ipython().run_cell_magic('time', '', 'cddds = encode_smiles(batch_list=smiles_list, npool=6)')


# In[11]:


cddds = np.concatenate(cddds)


# In[12]:


print(cddds.shape)


# #### Saving the cddd data 

# In[13]:


import h5py


# In[14]:


hf = h5py.File("../data/cddd.hdf5", "w")
hf.create_dataset("cddd", data=cddds)
hf.close()
print("Finished.")


# #### Stop the CDDD inference server execution in your other shell 

# In[ ]:




