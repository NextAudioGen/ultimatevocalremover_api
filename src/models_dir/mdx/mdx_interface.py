
from .tfc_tdf_v3 import TFC_TDF_net, STFT
from .tfc_tdf_v3 import TFC_TDF_net, STFT
from . import mdxnet as MdxnetSet
from . import spec_utils
from .constants import secondary_stem
from .constants import  MDX_NET_FREQ_CUT

import onnxruntime as ort
from onnx import load
from onnx2pytorch import ConvertModel

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

if torch.cuda.is_available(): device = "cuda"
elif torch.backends.mps.is_available(): device = torch.device("mps")
else: device = "cpu"


def load_mdx_models_data(model_path:str="mdx/modelparams/model_data.json")->dict:
    """
    Load the VR models data from the specified model path.

    Args:
        model_path (str): The path to the model data JSON file. Default is "mdx/modelparams/model_data.json".

    Returns:
        dict: The loaded models data.
    """

    models_data = json.load(open(model_path))
    return models_data

def get_model_hash_from_path(model_path:str="./mdx/weights/UVR-MDX-NET-Inst_1/UVR-MDX-NET-Inst_1.onnx")->str:
    """
    Get the hash of the model from the specified model path.

    Args:
        model_path (str): The path to the model file. Default is "./mdx/weights/UVR-MDX-NET-Inst_1/UVR-MDX-NET-Inst_1.onnx".

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

def load_from_ckpt(model_path:str, device:str):
    """
    Load a model from a checkpoint file and return the loaded model and its parameters.

    Args:
        model_path (str): The path to the checkpoint file.
        device (str): The device to load the model on.

    Returns:
        model_run (torch.nn.Module): The loaded model.
        (dim_c, hop) (tuple): The parameters of the model. (dim_c: int, hop: int)
    """
    model_params = torch.load(model_path, map_location=lambda storage, loc: storage)['hyper_parameters']
    dim_c, hop = model_params['dim_c'], model_params['hop_length']
    separator = MdxnetSet.ConvTDFNet(**model_params)
    model_run = separator.load_from_checkpoint(model_path).to(device).eval()
    return model_run, (dim_c, hop)

def device_to_ort_run_type(device:str):
    """
    Converts the device name to the corresponding ONNX Runtime execution provider run type.

    Args:
        device (str): The device name. Valid options are 'cuda', 'cpu', and any other value.

    Returns:
        list: A list containing the ONNX Runtime execution provider run type.

    """
    if device == 'cuda':
        run_type = ['CUDAExecutionProvider']
    else: #device == 'cpu':
        run_type = ['CPUExecutionProvider']
    # else:
    #     run_type = ['DnnlExecutionProvider']
    return run_type

def load_from_ort(model_path:str, device:str, segment_size:int, dim_t:int):

    if segment_size == dim_t and  device == 'cpu':
        ort_ = ort.InferenceSession(model_path, providers=device_to_ort_run_type(device))
        model_run = lambda spek:ort_.run(None, {'input': spek.cpu().numpy()})[0]
    else:
        model_run = ConvertModel(load(model_path))
        model_run.to(device).eval()
    
    return model_run, (4, 1024)

def load_modle(model_path:str, device:str='cuda', segment_size:int=None, dim_t:int=None):
    """
    Load the model from the given path and return the loaded model.

    Args:
        model_path (str): The path to the model file.
        device (str): The device to load the model on. Defaults to 'cuda'.
        segment_size (int): The segment size of the model. Defaults to None.
        dim_t (int): The time dimension of the model. Defaults to None.

    Returns:
        model_run (function): The loaded model.

    """
    if model_path.endswith('.onnx'):
        model_run = load_from_ort(model_path, device, segment_size, dim_t)
    else:
        model_run, (dim_c, hop) = load_from_ckpt(model_path, device)
    return model_run

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

def initialize_model_settings(n_fft:int, hop:int, dim_f:int, segment_size:int, device:str, **kwargs):
    n_bins = n_fft//2+1
    trim = n_fft//2
    chunk_size = hop * (segment_size-1)
    gen_size = chunk_size-2*trim
    stft = STFT(n_fft, hop, dim_f, device)

    return stft, n_bins, trim, chunk_size, gen_size

def run_model(mix, model_run, stft, adjust, denoise, device, is_match_mix):
    
    spek = stft(mix.to(device))*adjust
    spek[:, :, :3, :] *= 0 

    if is_match_mix:
        spec_pred = spek.cpu().numpy()
    else:
        if denoise:
            spec_pred = -model_run(-spek)*0.5+model_run(spek)*0.5  
        else:
            spec_pred = model_run(spek)

    return stft.inverse(torch.tensor(spec_pred).to(device)).cpu().detach().numpy()

def pitch_fix(source, sr_pitched, org_mix, semitone_shift):
    source = spec_utils.change_pitch_semitones(source, sr_pitched, semitone_shift=semitone_shift)[0]
    source = spec_utils.match_array_shapes(source, org_mix)
    return source

def demix(model_run, mix, prams, device='cpu', is_match_mix=False):
    
    semitone_shift = prams['semitone_shift']
    overlap = prams['overlap']
    adjust = prams['adjust']
    denoise = prams['denoise']
    org_mix = mix
    tar_waves_ = []
    stft, n_bins, trim, chunk_size, gen_size = initialize_model_settings(device=device, **prams)
    

    if is_match_mix:
        chunk_size = prams['hop'] * (256-1)
        overlap = 0.02
    else:
        chunk_size = chunk_size
        overlap = prams['overlap_mdx']
        
        if prams['semitone_shift']!=0:
            mix, sr_pitched = spec_utils.change_pitch_semitones(mix, 44100, semitone_shift=-prams['semitone_shift'])

    
    if semitone_shift:
        mix, sr_pitched = spec_utils.change_pitch_semitones(mix, 44100, semitone_shift=-semitone_shift)

    gen_size = chunk_size-2*trim

    pad = gen_size + trim - ((mix.shape[-1]) % gen_size)
    mixture = np.concatenate((np.zeros((2, trim), dtype='float32'), mix, np.zeros((2, pad), dtype='float32')), 1)

    if overlap is None:
        step = chunk_size - prams['n_fft']  
    else: 
        step = int((1 - overlap) * chunk_size)

    result = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
    divider = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
    total = 0
    total_chunks = (mixture.shape[-1] + step - 1) // step

    for i in range(0, mixture.shape[-1], step):
        total += 1
        start = i
        end = min(i + chunk_size, mixture.shape[-1])

        chunk_size_actual = end - start

        if overlap == 0:
            window = None
        else:
            window = np.hanning(chunk_size_actual)
            window = np.tile(window[None, None, :], (1, 2, 1))

        mix_part_ = mixture[:, start:end]
        if end != i + chunk_size:
            pad_size = (i + chunk_size) - end
            mix_part_ = np.concatenate((mix_part_, np.zeros((2, pad_size), dtype='float32')), axis=-1)

        mix_part = torch.tensor([mix_part_], dtype=torch.float32).to(device)
        mix_waves = mix_part.split(prams['mdx_batch_size'])
        
        with torch.no_grad():
            for mix_wave in mix_waves:

                tar_waves = run_model(mix_wave, model_run, stft, adjust, denoise, device, is_match_mix=is_match_mix)
                
                if window is not None:
                    tar_waves[..., :chunk_size_actual] *= window 
                    divider[..., start:end] += window
                else:
                    divider[..., start:end] += 1

                result[..., start:end] += tar_waves[..., :end-start]
        
    tar_waves = result / divider
    tar_waves_.append(tar_waves)

    tar_waves_ = np.vstack(tar_waves_)[:, :, trim:-trim]
    tar_waves = np.concatenate(tar_waves_, axis=-1)[:, :mix.shape[-1]]
    
    source = tar_waves[:,0:None]

    if semitone_shift:
        source = pitch_fix(source, sr_pitched, org_mix, semitone_shift)

    source =  source*prams['compensate']

    return source

def pitch_fix(source, sr_pitched, org_mix, semitone_shift):
        source = spec_utils.change_pitch_semitones(source, sr_pitched, semitone_shift=semitone_shift)[0]
        source = spec_utils.match_array_shapes(source, org_mix)
        return source
    
def match_frequency_pitch(mix, prams):
    source = mix
    semitone_shift = prams['semitone_shift']
    if prams['is_match_frequency_pitch'] and semitone_shift!=0:
        source, sr_pitched = spec_utils.change_pitch_semitones(mix, 44100, semitone_shift=-semitone_shift)
        source = pitch_fix(source, sr_pitched, mix, semitone_shift)

    return source

def get_secondery_stems(model_run, source, mix, prams, device='cpu'):
    mdx_net_cut = False

    if (prams['primary_stem'] in MDX_NET_FREQ_CUT) and prams['is_match_frequency_pitch']:
        mdx_net_cut = True

    if mdx_net_cut:
        raw_mix = demix(model_run, match_frequency_pitch(mix, prams), prams, device=device, is_match_mix=True)  
    else:
        match_frequency_pitch(mix, prams)

    if prams['is_invert_spec']:
        secondary_source = spec_utils.invert_stem(raw_mix, source) 
    else: 
        secondary_source = mix.T-source.T
    
    return secondary_source

def nparray_stem_to_dict(stems, second_stem, model_data):
    if stems.shape[0] != 2:
        stems = stems.T
    if second_stem.shape[0] != 2:
        second_stem = second_stem.T
    return {
        model_data['primary_stem'].lower(): stems,
        secondary_stem(model_data['primary_stem']).lower(): second_stem
    }

if __name__ == "__main__":
    models_data = load_mdx_models_data(model_path="mdx/modelparams/model_data.json")
    model_hash = get_model_hash_from_path(model_path="./mdx/weights/UVR-MDX-NET-Inst_1/UVR-MDX-NET-Inst_1.onnx")

    model_data = models_data[model_hash]

    segment_size = 256
    dim_t = model_data['mdx_dim_t_set']
    model_run, (dim_c, hop) = load_modle('./mdx/weights/UVR-MDX-NET-Inst_1/UVR-MDX-NET-Inst_1.onnx', device, segment_size=segment_size, dim_t=dim_t)

    audio_file = "/Users/mohannadbarakat/Downloads/t.wav"
    mix = prepare_mix(audio_file)

    prams = {
        'n_fft': model_data['mdx_n_fft_scale_set'],
        'hop': hop,
        'dim_f': model_data['mdx_dim_f_set'],
        'segment_size': 256,
        'overlap': 0.75,
        'mdx_batch_size': 1,
        'semitone_shift': 0,
        'compensate': model_data['compensate'],
        'adjust': 1.08, 
        'denoise': False,
        'is_invert_spec': False,
        'primary_stem': model_data['primary_stem'],
        'is_match_frequency_pitch': True,
        'overlap_mdx': None
    }

    stems = demix(mix, prams, device=device)

    second_stem = get_secondery_stems(stems, mix, prams, device='cpu')

    dect_stems = nparray_stem_to_dict(stems, second_stem, model_data)


