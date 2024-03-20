
from . import nets_new
from . import nets
from . import spec_utils

import torch
import json
import hashlib
import librosa
import numpy as np
import audioread
import platform
from numpy.typing import NDArray
from typing import Union
import math
import os
from .constants import NON_ACCOM_STEMS, VOCAL_STEM, OPERATING_SYSTEM, SYSTEM_PROC, SYSTEM_ARCH, ARM, N_BINS
import pathlib
from pathlib import Path


class ModelParameters(object):
    """
    This class is used to store the parameters of the model. It reads the configuration file and stores the parameters in the instance.
    """
    def __init__(self, config_path:str=''):
        """
        Initializes an instance of ModelParameters.It reads the configuration file and stores the parameters in the instance.
        all the parameters are stored in the self.param dictionary.

        Args:
            config_path (str): The path to the configuration file.

        """
        config_path = os.path.join(pathlib.Path(__file__).parent.resolve(), config_path)
        with open(config_path, 'r') as f:
                self.param = json.loads(f.read(), object_pairs_hook=int_keys)
                
        for k in ['mid_side', 'mid_side_b', 'mid_side_b2', 'stereo_w', 'stereo_n', 'reverse']:
            if not k in self.param:
                self.param[k] = False
                
        if N_BINS in self.param:
            self.param['bins'] = self.param[N_BINS]

def load_vr_models_data(model_path:str="./modelparams/model_data.json")->dict:
    """
    Load the VR models data from the specified model path.

    Args:
        model_path (str): The path to the model data JSON file. Default is "./modelparams/model_data.json".

    Returns:
        dict: The loaded models data.
    """
    # model_path = os.path.join(pathlib.Path(__file__).parent.resolve(), model_path)
    models_data = json.load(open(model_path))
    return models_data

def get_model_hash_from_path(model_path:str="./weights/1_HP-UVR.pth")->str:
    """
    Get the hash of the model from the specified model path.

    Args:
        model_path (str): The path to the model file. Default is "./weights/1_HP-UVR.pth".

    Returns:
        str: The hash of the model.
    """
    # model_path = os.path.join(pathlib.Path(__file__).parent.resolve(), model_path)

    try:
        with open(model_path, 'rb') as f:
            f.seek(- 10000 * 1024, 2)
            model_hash = hashlib.md5(f.read()).hexdigest()
    except:
        model_hash = hashlib.md5(open(model_path,'rb').read()).hexdigest()
    
    return model_hash

def int_keys(d):
    """
    Args:
        d (dict): The input dictionary.

    Returns:
        dict: A new dictionary with the keys converted to integers if they are numeric strings.

    Example:
        >>> d = {'1': 'one', '2': 'two', '3': 'three'}
        >>> int_keys(d)
        {1: 'one', 2: 'two', 3: 'three'}
        
        The int_keys function is a helper function defined outside the ModelParameters class. 
        It takes a dictionary d as input and returns a new dictionary r where the keys are converted 
        to integers if they are numeric strings.
    """
    r = {}
    for k, v in d:
        if k.isdigit():
            k = int(k)
        r[k] = v
    return r

def get_capacity_and_vr_model(model_data)->tuple:
    """
    Get the capacity and VR model from the specified model data.

    Args:
        model_data (dict): The model data.

    Returns:
        tuple: The VR model and capacity. (is_vr_51_model, model_capacity)
    
    The variable is_vr_51_model is a boolean flag that indicates whether the VR model is a 5.1 vocal remover model.
    It is used to determine the capacity of the model. If is_vr_51_model is True, 
    then the model capacity is determined by the values of model_data["nout"] and model_data["nout_lstm"]. 
    Otherwise, the default model capacity is set to (32, 128).
    """

   
    is_vr_51_model = False
    model_capacity = 32, 128

    if "nout" in model_data.keys() and "nout_lstm" in model_data.keys():
        model_capacity = model_data["nout"], model_data["nout_lstm"]
        is_vr_51_model = True

    return is_vr_51_model, model_capacity

def _load_model_with_hprams(model_path:str, mp:ModelParameters, 
                            is_vr_51_model:bool=False, model_capacity:tuple=(32, 128),
                            device:str="cuda")->torch.nn.Module:
    """
    Loads a model from the given model_path and returns the loaded model.

    Parameters:
        model_path (str): The path to the model file.
        mp (ModelParameters): An instance of ModelParameters class.
        is_vr_51_model (bool): Indicates whether the model is a VR 5.1 model.
        model_capacity (tuple): A tuple representing the model capacity. Default is (32, 128).
        device (str): The device to load the model on. Default is "cuda".

    Returns:
        model_run: The loaded model.
    """
    nn_arch_sizes = [
        31191, # default
        33966, 56817, 123821, 123812, 129605, 218409, 537238, 537227]
    vr_5_1_models = [56817, 218409]
    model_size = math.ceil(os.stat(model_path).st_size / 1024)
    nn_arch_size = min(nn_arch_sizes, key=lambda x:abs(x-model_size))

    if nn_arch_size in vr_5_1_models or is_vr_51_model:
        model_run = nets_new.CascadedNet(mp.param['bins'] * 2, 
                                                nn_arch_size, 
                                                nout=model_capacity[0], 
                                                nout_lstm=model_capacity[1])
        
    else:
        model_run = nets.determine_model_capacity(mp.param['bins'] * 2, nn_arch_size)
                    
    model_run.load_state_dict(torch.load(model_path, map_location='cpu')) 
    model_run.to(device) 

    return model_run

def get_secondary_stem(primary_stem:str)->str:
    """
    Get the secondary stem from the given primary stem.

    Args:
        primary_stem (str): The primary stem.

    Returns:
        str: The secondary stem.
    """
    stem_couples = {
        'Vocals': 'Instrumental',
        'Instrumental': 'Vocals',
    }
    
    if primary_stem in stem_couples:
        return stem_couples[primary_stem]
    else:
        if 'no' in primary_stem.lower(): return primary_stem.replace('no', '')
        else: return 'no' + primary_stem

def load_model(model_path:str, device:str="cuda")->tuple:
    """
    Loads a model from the given model path.

    Args:
        model_path (str): The path to the model file.
        device (str): The device to load the model on. Default is "cuda".

    Returns:
        torch.nn.Module: The loaded model, 
        ModelParameters: Model parameters, 
        bool: A boolean flag indicating whether the model is a 5.1 vocal remover model,
        dict(str, str): Model stems names (e.g. {"primary_stem": "vocals", "secondary_stem": "instruments"}).
    """
    
    model_hash = get_model_hash_from_path(model_path)
    model_data = MODELS_DATA[model_hash]

    mp = f"./modelparams/{model_data['vr_model_param']}.json"

    mp = ModelParameters(mp)

    is_vr_51_model, model_capacity = get_capacity_and_vr_model(model_data)

    model_run = _load_model_with_hprams(model_path, mp, is_vr_51_model, model_capacity, device)

    primary_stem = model_data['primary_stem']
    secondary_stem = get_secondary_stem(primary_stem)
    strems = {"primary_stem":primary_stem, "secondary_stem":secondary_stem}
    return model_run, mp, is_vr_51_model, strems

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

def loading_mix(audio_file:Union[str, NDArray], mp:ModelParameters, is_vr_51_model:bool,
                 wav_type_set:str="PCM_U8", high_end_process=None, ):
    """
    Load and process the audio mix.

    Parameters:
        audio_file (str or numpy.ndarray): Path to the audio file or audio data as a numpy array.
        mp (object): Object containing parameters for audio processing.
        is_vr_51_model (bool): Flag indicating whether the model is a 5.1 vocal remover model.
        wav_type_set (str): Subtype of the audio file. Options are ('PCM_U8', 'PCM_16', 'PCM_24', 'PCM_32', '32-bit Float', '64-bit Float') Default is "PCM_U8".
        high_end_process (None or str): Type of high-end processing to be applied.

    Returns:
        X_spec (numpy.ndarray): Combined spectrogram of the audio mix.
        input_high_end (numpy.ndarray or None): High-end portion of the audio mix.
        input_high_end_h (int or None): Height of the high-end portion of the audio mix.
    """
    X_wave, X_spec_s = {}, {}
    
    bands_n = len(mp.param['band'])
    if OPERATING_SYSTEM == 'Darwin':
        if SYSTEM_PROC == ARM or ARM in SYSTEM_ARCH:
            wav_resolution = 'polyphase'  
        else:
             wav_resolution = "soxr_hq"
    else:
        wav_resolution = "soxr_hq"
    # audio_file = spec_utils.write_array_to_mem(audio_file, subtype=wav_type_set)
    
    is_mp3 = False
    if isinstance(audio_file, str):
        is_mp3 = audio_file.endswith('.mp3') 
        if is_mp3:
            audio_file = rerun_mp3(audio_file)
        else:
            sr = mp.param['band'][bands_n]['sr']
            audio_file, _ = librosa.load(audio_file, sr=sr, mono=False, dtype=np.float32, res_type=wav_resolution)
    
    if isinstance(audio_file, torch.Tensor):
        audio_file = audio_file.numpy()

    for d in range(bands_n, 0, -1):        
        bp = mp.param['band'][d]
    
        if d == bands_n: # high-end band
            X_wave[d] = audio_file
            X_spec_s[d] = spec_utils.wave_to_spectrogram(X_wave[d], bp['hl'], bp['n_fft'], mp, band=d, is_v51_model=is_vr_51_model)
                
            # make sure it is stereo
            if X_wave[d].ndim == 1: X_wave[d] = np.asarray([X_wave[d], X_wave[d]])

        else: # lower bands
            X_wave[d] = librosa.resample(X_wave[d+1], orig_sr=mp.param['band'][d+1]['sr'], target_sr=bp['sr'], res_type=wav_resolution)
            X_spec_s[d] = spec_utils.wave_to_spectrogram(X_wave[d], bp['hl'], bp['n_fft'], mp, band=d, is_v51_model=is_vr_51_model)

        if d == bands_n and (high_end_process is not None):
            input_high_end_h = (bp['n_fft']//2 - bp['crop_stop']) + (mp.param['pre_filter_stop'] - mp.param['pre_filter_start'])
            input_high_end = X_spec_s[d][:, bp['n_fft']//2-input_high_end_h:bp['n_fft']//2, :]
        else:
            input_high_end_h = input_high_end = None
            
    X_spec = spec_utils.combine_spectrograms(X_spec_s, mp, is_v51_model=is_vr_51_model)
    
    del X_wave, X_spec_s, audio_file

    return X_spec, input_high_end, input_high_end_h

def _execute(X_mag_pad:NDArray, roi_size:int, model_run:torch.nn.Module, 
             batch_size:int, is_tta:bool, window_size:int, device:str='cuda')->NDArray:
    """
    Executes the vocal removal algorithm on the given input spectrogram.

    Args:
        X_mag_pad (ndarray): Input spectrogram with padding.
        roi_size (int): Size of the region of interest.
        model_run (object): Instance of the model to run.
        batch_size (int): Batch size for processing.
        is_tta (bool): Flag indicating whether to use test-time augmentation.
        window_size (int): Size of the sliding window.
        device (str): Device to run the model on. Default is "cuda".

    Returns:
        ndarray: Output mask representing the vocal component of the input spectrogram.
    """
    X_dataset = []
    patches = (X_mag_pad.shape[2] - 2 * model_run.offset) // roi_size
    total_iterations = patches//batch_size if not is_tta else (patches//batch_size)*2
    for i in range(patches):
        start = i * roi_size
        X_mag_window = X_mag_pad[:, :, start:start + window_size]
        X_dataset.append(X_mag_window)

    X_dataset = np.asarray(X_dataset)
    model_run.eval()
    with torch.no_grad():
        mask = []
        for i in range(0, patches, batch_size):
            
            X_batch = X_dataset[i: i + batch_size]
            X_batch = torch.from_numpy(X_batch).to(device)
            pred = model_run.predict_mask(X_batch)
            
            pred = pred.detach().cpu().numpy()
            pred = np.concatenate(pred, axis=2)
            mask.append(pred)
        mask = np.concatenate(mask, axis=2)
    return mask

def postprocess(mask:NDArray, X_mag:NDArray, X_phase:NDArray, primary_stem:str, 
                aggressiveness:float=.05, post_process_threshold:Union[None, float]=None):
    """
    Post-processes the mask to obtain the separated vocal and instrumental spectrograms.

    Args:
        mask (ndarray): The binary mask indicating the presence of vocals in the mixture.
        X_mag (ndarray): The magnitude spectrogram of the mixture.
        X_phase (ndarray): The phase spectrogram of the mixture.
        primary_stem (str): The primary stem to be separated (e.g., 'vocals', 'drums', 'bass', etc.).
        aggressiveness (float): The aggressiveness parameter for adjusting the mask. Value should be between -1 and 1. Default is 0.05 (best for vocals).
        post_process_threshold (float, optional): The threshold for merging artifacts in the mask. If None, no post-processing is applied. Default is None.

    Returns:
        y_spec (ndarray): The separated vocal spectrogram.
        v_spec (ndarray): The separated instrumental spectrogram.
    """
    is_non_accom_stem = False
    for stem in NON_ACCOM_STEMS:
        if stem == primary_stem:
            is_non_accom_stem = True
    

    mask = spec_utils.adjust_aggr(mask, is_non_accom_stem, aggressiveness)

    if post_process_threshold is not None:
        mask = spec_utils.merge_artifacts(mask, thres=post_process_threshold)

    y_spec = mask * X_mag * np.exp(1.j * X_phase)
    v_spec = (1 - mask) * X_mag * np.exp(1.j * X_phase)

    return y_spec, v_spec
    
def inference_vr(model_run, X_spec:NDArray, aggressiveness:float=.05, 
                  window_size:int=512, is_tta:bool=False, batch_size:int=4, 
                  post_process_threshold:Union[None, float]=None, 
                  primary_stem:str=VOCAL_STEM, device:str='cuda')->tuple:
    """
    Perform vocal removal inference on a given spectrogram.

    Args:
        model_run (object): The model to run.
        X_spec (ndarray): The input spectrogram.
        aggressiveness (float): The aggressiveness parameter for adjusting the mask. Value should be between -1 and 1. Default is 0.05 (best for vocals).
        window_size (int): The size of the window for processing.
        is_tta (bool): Flag indicating whether to use test-time augmentation.
        batch_size (int): Batch size for processing.
        post_process_threshold (float, optional): The threshold for merging artifacts in the mask. If None, no post-processing is applied. Default is None.
        primary_stem (str): The primary stem to be separated (e.g., 'vocals', 'drums', 'bass', etc.).

    Returns:
        tuple: The separated vocal and instrumental spectrograms. (y_spec:NDarray, v_spec:NDarray)
    """

    X_mag, X_phase = spec_utils.preprocess(X_spec)
    n_frame = X_mag.shape[2]
    pad_l, pad_r, roi_size = spec_utils.make_padding(n_frame, window_size, model_run.offset)
    X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
    X_mag_pad /= X_mag_pad.max()
    mask = _execute(X_mag_pad, roi_size, model_run, batch_size, is_tta, window_size, device=device)

    if is_tta:
        pad_l += roi_size // 2
        pad_r += roi_size // 2
        X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_mag_pad /= X_mag_pad.max()
        mask_tta = _execute(X_mag_pad, roi_size, model_run, batch_size, is_tta, window_size, device=device)
        mask_tta = mask_tta[:, :, roi_size // 2:]
        mask = (mask[:, :, :n_frame] + mask_tta[:, :, :n_frame]) * 0.5
    else:
        mask = mask[:, :, :n_frame]

    y_spec, v_spec = postprocess(mask, X_mag, X_phase, primary_stem, aggressiveness, post_process_threshold)
    
    return y_spec, v_spec

def convert_spec_to_wav(spec:NDArray, model_params:ModelParameters, is_vr_51_model:bool,
                         high_end_process:str, input_high_end:NDArray, input_high_end_h:bool)->NDArray:    
    """
    Converts a spectrogram to a waveform.

    Args:
        spec (np.ndarray): The input spectrogram.
        model_params (ModelParameters): The parameters of the model.
        is_vr_51_model (bool): Indicates whether the model is a 5.1 model.
        high_end_process (str): The high-end processing method.
        input_high_end (np.ndarray): The input high-end data.
        input_high_end_h (bool): Indicates whether the input high-end data is available.

    Returns:
        np.ndarray: The converted waveform.
    """
    if isinstance(high_end_process, str) and high_end_process.startswith('mirroring') and isinstance(input_high_end, np.ndarray) and input_high_end_h:        
        input_high_end_ = spec_utils.mirroring(high_end_process, spec, input_high_end, model_params)
        wav = spec_utils.cmb_spectrogram_to_wave(spec, model_params, input_high_end_h, input_high_end_, is_v51_model=is_vr_51_model)
        wav = spec_utils.cmb_spectrogram_to_wave(spec, model_params, input_high_end_h, input_high_end_, is_v51_model=is_vr_51_model)
    else:
        wav = spec_utils.cmb_spectrogram_to_wave(spec, model_params, is_v51_model=is_vr_51_model)
        
    return wav

def convert_audio_spec_to_wav(spec:NDArray, model_params:ModelParameters, 
                              is_vr_51_model:bool, model_samplerate:int,
                              high_end_process:bool, input_high_end:bool, input_high_end_h:bool)->NDArray:
    """
    Convert audio spectrogram to waveform.

    Args:
        spec (numpy.ndarray): Input audio spectrogram.
        model_params (ModelParameters): Model parameters.
        is_vr_51_model (bool): Flag indicating if the model is a 5.1 model.
        model_samplerate (int): Model sample rate.
        high_end_process (bool): Flag indicating if high-end processing is enabled.
        input_high_end (bool): Flag indicating if high-end data is available.
        input_high_end_h (bool): Flag indicating if high-end data height is available.

    Returns:
        numpy.ndarray: Output audio waveform.
    """
    res = convert_spec_to_wav(spec, model_params, is_vr_51_model, high_end_process, input_high_end, input_high_end_h).T
    if model_samplerate == 44100:
        res = librosa.resample(res.T, orig_sr=model_samplerate, target_sr=44100).T
    return res.T

def get_audio(y_spec:NDArray, v_spec:NDArray, model_params:ModelParameters,
               normaliz:bool, is_vr_51_model:bool,
               high_end_process:bool, input_high_end:bool, input_high_end_h:bool)->dict:
    """
    Convert audio spectrograms to audio waveforms and normalize them.

    Args:
        y_spec (numpy.ndarray): Spectrogram of the primary source audio.
        v_spec (numpy.ndarray): Spectrogram of the secondary source audio.
        model_params (ModelParams): Parameters of the model.
        normaliz (bool): Flag indicating whether to normalize the audio waveforms.
        is_vr_51_model (bool): Flag indicating whether the model is a 5.1 model.
        high_end_process (bool): Flag indicating if high-end processing is enabled.
        input_high_end (bool): Flag indicating if high-end data is available.
        input_high_end_h (bool): Flag indicating if high-end data height is available.

    Returns:
        dict: A dictionary containing the primary source audio waveform and the secondary source audio waveform.
    """
    model_samplerate = model_params.param['sr']
    primary_source = convert_audio_spec_to_wav(y_spec, model_params, is_vr_51_model, model_samplerate, high_end_process, input_high_end, input_high_end_h)
    secondary_source = convert_audio_spec_to_wav(v_spec, model_params, is_vr_51_model, model_samplerate, high_end_process, input_high_end, input_high_end_h)
    
    primary_source = spec_utils.normalize(primary_source, normaliz)
    secondary_source = spec_utils.normalize(secondary_source, normaliz)

    return {"primary_stem": primary_source, 
            "secondary_stem": secondary_source}

def rename_audio_res_dict(audio_res:dict, names:dict)->dict:
    """
    Rename the keys of the audio results dictionary.

    Args:
        audio_res (dict): The audio results dictionary.
        names (dict): A dictionary containing the new names for the audio sources.

    Returns:
        dict: The renamed audio results dictionary.
    """
    primary_name = names["primary_stem"]
    secondary_name = names["secondary_stem"]
    audio_res = {primary_name: audio_res["primary_stem"], 
                 secondary_name: audio_res["secondary_stem"]}
    
    audio_res = {k.lower(): v for k, v in audio_res.items()}
    return audio_res

def get_audio_dict(y_spec:NDArray, v_spec:NDArray, stems:dict, 
                   model_params:ModelParameters, normaliz:bool, is_vr_51_model:bool,
                   high_end_process:bool, input_high_end:bool, input_high_end_h:bool)->dict:
    """
    Convert audio spectrograms to audio waveforms and normalize them.

    Args:
        y_spec (numpy.ndarray): Spectrogram of the primary source audio.
        v_spec (numpy.ndarray): Spectrogram of the secondary source audio.
        model_params (ModelParameters): Parameters of the model.
        names (dict): A dictionary containing the new names for the audio sources.
        normaliz (bool): Flag indicating whether to normalize the audio waveforms.
        is_vr_51_model (bool): Flag indicating whether the model is a 5.1 model.
        high_end_process (bool): Flag indicating if high-end processing is enabled.
        input_high_end (bool): Flag indicating if high-end data is available.
        input_high_end_h (bool): Flag indicating if high-end data height is available.

    Returns:
        dict: A dictionary containing the primary source audio waveform and the secondary source audio waveform.
    """
    audio_res = get_audio(y_spec, v_spec, model_params, normaliz, is_vr_51_model, high_end_process, input_high_end, input_high_end_h)
    audio_res = rename_audio_res_dict(audio_res, stems)
    return audio_res

uvr_path = Path(__file__).parent.parent.parent
MODELS_DATA = load_vr_models_data(os.path.join(uvr_path, "models_dir", "vr_network", "modelparams", "model_data.json"))

if __name__ == "__main__":
    import audiofile
    device = torch.device("mps")
    model_path = ""
    audio_file = ""
    high_end_process = None
    wav_type_set = 'PCM_U8'
    window_size = 512
    post_process_threshold = None
    batch_size = 4 
    is_tta=False
    aggressiveness = .05
    normaliz = False
    high_end_process=False
    input_high_end=False
    input_high_end_h=False

    model_run, mp, is_vr_51_model, stems  = load_model(model_path, device)


    aggressiveness = {'value': aggressiveness, 
                    'split_bin': mp.param['band'][1]['crop_stop'], 
                    'aggr_correction': mp.param.get('aggr_correction')}

    inp, input_high_end, input_high_end_h = loading_mix(audio_file, mp, is_vr_51_model, 
                                                        wav_type_set=wav_type_set, high_end_process=high_end_process)


    y_spec, v_spec = inference_vr(X_spec=inp, 
                                aggressiveness=aggressiveness, 
                                window_size=window_size,
                                model_run=model_run, 
                                is_tta=is_tta, 
                                batch_size=batch_size, 
                                post_process_threshold=post_process_threshold, 
                                primary_stem=stems["primary_stem"])


    audio_res = get_audio_dict(y_spec=y_spec, v_spec=v_spec, stems=stems, model_params=mp,
                                normaliz=normaliz, is_vr_51_model=is_vr_51_model,
                                high_end_process=high_end_process, input_high_end=input_high_end, 
                                input_high_end_h=input_high_end_h)

    
    audio_res.keys()

    
    model_samplerate = mp.param['sr']

    vocals_path = "vocals.wav"
    insturemntal_path = "insturemntas.wav"

    audiofile.write(vocals_path, audio_res["Vocals"], model_samplerate)
    audiofile.write(insturemntal_path, audio_res["Instrumental"], model_samplerate)
