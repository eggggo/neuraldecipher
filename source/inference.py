from turtle import forward
from cddd.inference import InferenceServer
import torch
import numpy as np
import os
import h5py
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
import rdkit.Chem as Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import os
import json
from multiprocessing import Pool
from models import Neuraldecipher
from utils import get_loader_list

inference_server = InferenceServer(port_frontend=5527, use_running=True)

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


def forward_pass_dataloader(dataloader, device, neuraldecipher):
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
        for batch in dataloader:
                ecpf_in = batch.to(device=device, dtype=torch.float32)
                output = neuraldecipher(ecpf_in) 
                predicted_cddd.append(output.detach().cpu().numpy())

    predicted_cddd = np.concatenate(predicted_cddd)

    return predicted_cddd


def get_batches_from_large_list(large_list, batch_size):
    n_batches = len(large_list) // batch_size
    rest_indices = len(large_list) - n_batches*batch_size
    last_start = n_batches*batch_size
    last_end = last_start + rest_indices
    batches = [large_list[i*batch_size:(i+1)*batch_size] for i in range(n_batches)]
    batches.append(large_list[last_start:last_end])
    return batches


def canonicalize_sanitize_smiles(smiles, sanitize=True):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if sanitize:
            Chem.SanitizeMol(mol)
        smi = Chem.MolToSmiles(mol)
        return smi
    except:
        None


def decode_cddd(batch_list, npool=6):
    with Pool(npool) as pool:
        decoded_smiles = pool.map(inference_server.emb_to_seq, batch_list)
    return decoded_smiles


def predict(ecfp_list):
    # input dimesnions of (n, 1024), list of 1024 bit ecfps
    # load neuraldecipher model
    neuraldecipher = load_model("../params/1024_config_count_gpu.json",
                            "../models/1024_final_model_cs_gpu/weights.pt")

    device = torch.device("cuda:0")
    neuraldecipher = neuraldecipher.eval()    
    neuraldecipher = neuraldecipher.to(device)

    # load data into torch dataloader for batching
    ecfp_list = np.array(ecfp_list)
    tensor_ecfp_list = torch.Tensor(ecfp_list)
    ecfp_dataloader = get_loader_list(tensor_ecfp_list)
    
    # perform neuraldecipher cddd forward pass prediction and bacth
    print('predicting cddds')
    pred_cddd = forward_pass_dataloader(ecfp_dataloader, device, neuraldecipher)
    pred_cddd = get_batches_from_large_list(pred_cddd, 1024)

    # decode cddd results using cddd to smiles
    print('decoding cddds')
    decoded_smiles = decode_cddd(pred_cddd)
    decoded_smiles = np.concatenate(decoded_smiles)

    # canonicalize smiles (invalid smiles will end as None)
    canonical_smiles = [canonicalize_sanitize_smiles(s, sanitize=True) for s in decoded_smiles]

    return canonical_smiles