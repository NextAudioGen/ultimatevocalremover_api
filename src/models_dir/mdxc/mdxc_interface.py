
from .tfc_tdf_v3 import TFC_TDF_net, STFT
from .tfc_tdf_v3 import TFC_TDF_net, STFT
from  . import mdxnet as MdxnetSet
from . import spec_utils
from .constants import secondary_stem
import onnxruntime as ort
from onnx import load
from onnx2pytorch import ConvertModel
from ml_collections import ConfigDict

import torch
import audiofile
import soundfile as sf
import json 
import hashlib
import librosa
import numpy as np
import audioread
import platform
from numpy.typing import NDArray
from typing import Union
import math, os
import yaml

if torch.cuda.is_available(): device = "cuda"
elif torch.backends.mps.is_available(): device = torch.device("mps")
else: device = "cpu"


def load_mdxc_models_data(model_path:str="mdxc/modelparams/model_data.json")->dict:
    """
    Load the mdxc models data from the specified model path.

    Args:
        model_path (str): The path to the model data JSON file. Default is "mdxc/modelparams/model_data.json".

    Returns:
        dict: The loaded models data.
    """

    models_data = json.load(open(model_path))
    return models_data

def get_model_hash_from_path(model_path:str="./mdxc/weights/MDX23C-8KFFT-InstVoc_HQ/MDX23C-8KFFT-InstVoc_HQ.ckpt")->str:
    """
    Get the hash of the model from the specified model path.

    Args:
        model_path (str): The path to the model file. Default is "./mdxc/weights/UVR-MDX-NET-Inst_1/UVR-MDX-NET-Inst_1.ckpt".

    Returns:
        str: The hash of the model.
    """

    try:
        with open(model_path, 'rb') as f:
            f.seek(- 10000 * 1024, 2)
            model_hash = hashlib.md5(f.read()).hexdigest()
    except:
        model_hash = hashlib.md5(open(model_path,'rb').read()).hexdigest()
    
    return model_hash

def load_mdxc_model_data(models_data, model_hash, model_path="./mdxc/modelparams")->ConfigDict:
    """
    Load the mdxc model data from the specified models data and model hash.

    Args:
        models_data (dict): The models data.
        model_hash (str): The hash of the model.

    Returns:
        dict: The loaded model data.
    """

    model_data_src = models_data[model_hash]
    # if not "config_yaml" in model_data_src: return model_data_src
    model_path = os.path.join(model_path, "mdx_c_configs", model_data_src['config_yaml'])
    model_data = yaml.load(open(model_path), Loader=yaml.FullLoader)

    model_data = ConfigDict(model_data)
    
    return model_data

def load_modle(model_path:str, model_data:ConfigDict, device:str='cuda')->torch.nn.Module:
    """
    Load the model from the given path and return the loaded model.

    Args:
        model_path (str): The path to the model file.
        model_data (ConfigDict): The model data.
        device (str): The device to load the model on. Defaults to 'cuda'.

    Returns:
        model_run (function): The loaded model.

    """
    model = TFC_TDF_net(model_data, device=device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device).eval()
    return model

def rerun_mp3(audio_file:NDArray, sample_rate:int=44100):
    """
    Load an audio file and return the audio data.

    Parameters:
        audio_file (str): The path to the audio file.
        sample_rate (int, optional): The desired sample rate of the audio data. Default is 44100.

    Returns:
        numpy.ndarray: The audio data as a numpy array.
    """
    with audioread.audio_open(audio_file) as f:
        track_length = int(f.duration)

    return librosa.load(audio_file, duration=track_length, mono=False, sr=sample_rate)[0]

def prepare_mix(mix):
    
    audio_path = mix

    if not isinstance(mix, np.ndarray):
        mix, sr = librosa.load(mix, mono=False, sr=44100)
    else:
        mix = mix

    if isinstance(audio_path, str):
        if not np.any(mix) and audio_path.endswith('.mp3'):
            mix = rerun_mp3(audio_path)

    if mix.ndim == 1:
        mix = np.asfortranarray([mix,mix])

    return mix

def pitch_fix(source, sr_pitched, org_mix, semitone_shift)->np.ndarray:
    source = spec_utils.change_pitch_semitones(source, sr_pitched, semitone_shift=semitone_shift)[0]
    source = spec_utils.match_array_shapes(source, org_mix)
    return source

def demix(mix:np.ndarray, prams:dict, model:torch.nn.Module, model_data:ConfigDict, device:str='cpu')->dict:
    """
    Demixes the input audio mixture into its constituent sources using a given model.

    Args:
        mix (np.ndarray): The input audio mixture.
        prams (dict): A dictionary containing various parameters for demixing.
        model (torch.nn.Module): The demixing model.
        model_data (ConfigDict): The configuration data for the model.
        device (str, optional): The device to run the demixing on. Defaults to 'cpu'.

    Returns:
        dict: A dictionary containing the estimated sources.

    """
    
    sr_pitched = 441000
    org_mix = mix
    semitone_shift = prams['semitone_shift']
    if  semitone_shift != 0:
        mix, sr_pitched = spec_utils.change_pitch_semitones(mix, 44100, semitone_shift=-semitone_shift)

    
    mix = torch.tensor(mix, dtype=torch.float32)

    try:
        S = model.num_target_instruments
    except Exception as e:
        S = model.module.num_target_instruments

    if prams['is_mdx_c_seg_def']:
        mdx_segment_size = model_data.inference.dim_t  
    else:
        mdx_segment_size = prams['segment_size']
    
    batch_size = prams['batch_size']
    chunk_size = model_data.audio.hop_length * (mdx_segment_size - 1)
    overlap = prams['overlap_mdx23']

    hop_size = chunk_size // overlap
    mix_shape = mix.shape[1]
    pad_size = hop_size - (mix_shape - chunk_size) % hop_size
    mix = torch.cat([torch.zeros(2, chunk_size - hop_size), mix, torch.zeros(2, pad_size + chunk_size - hop_size)], 1)

    chunks = mix.unfold(1, chunk_size, hop_size).transpose(0, 1)
    batches = [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]
    
    X = torch.zeros(S, *mix.shape) if S > 1 else torch.zeros_like(mix)
    X = X.to(device)

    with torch.no_grad():
        cnt = 0
        for batch in batches:
            x = model(batch.to(device))
            
            for w in x:
                X[..., cnt * hop_size : cnt * hop_size + chunk_size] += w
                cnt += 1

    estimated_sources = X[..., chunk_size - hop_size:-(pad_size + chunk_size - hop_size)] / overlap
    del X
    pitch_fix = lambda s:pitch_fix(s, sr_pitched, org_mix, semitone_shift)

    if S > 1:
        sources = {k: pitch_fix(v) if semitone_shift!=0 else v for k, v in zip(model_data.training.instruments, estimated_sources.cpu().detach().numpy())}
        del estimated_sources   
        return sources
    
    est_s = estimated_sources.cpu().detach().numpy()
    del estimated_sources

    if semitone_shift!=0:
        return pitch_fix(est_s)  
    else:
        return est_s

def rename_stems(stems:dict)->dict:
    """
    Basicly, applay .lower() to all keys in the stems dict.
    Args:
        stems (dict): The stems to be renamed.
        model_data (ConfigDict): The model data.

    Returns:
        dict: The renamed stems.
    """

    return {k.lower():v for k,v in stems.items()}


if __name__ == "__main__":
    models_data = load_mdxc_models_data(model_path="mdxc/modelparams/model_data.json")
    model_hash = get_model_hash_from_path(model_path="./mdxc/weights/MDX23C-8KFFT-InstVoc_HQ/MDX23C-8KFFT-InstVoc_HQ.ckpt")

    model_data = models_data[model_hash]

    model_data = load_mdxc_model_data(models_data, model_hash, model_path="./mdxc/modelparams")


    model_data
    model_run = load_modle("./mdxc/weights/MDX23C-8KFFT-InstVoc_HQ/MDX23C-8KFFT-InstVoc_HQ.ckpt",
                        model_data, device)
    audio_file = "/Users/mohannadbarakat/Downloads/t.wav"
    mix = prepare_mix(audio_file)
    mix.shape

    segment_size = 256
    prams = {
        'is_mdx_c_seg_def': False,
        'segment_size': segment_size,
        'batch_size': 1,
        'overlap_mdx23': 8,
        'semitone_shift': 0,
        # 'mdx_segment_size': segment_size
    }

    stems = demix(mix, prams, model_run, model_data, device)

    stems = rename_stems(stems)
    stems.keys()

