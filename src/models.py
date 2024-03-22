import types

from numpy._typing import NDArray
from .utils.get_models import download_model, model_exists
import json 
from .models_dir.demucs.demucs import api as demucs_api
from .models_dir.demucs import demucs 
from .models_dir.vr_network import vr_interface as vr_api
from .models_dir.mdx import mdx_interface as mdx_api
from .models_dir.mdxc import mdxc_interface as mdxc_api

import torch
import os
from pathlib import Path
import sys
import numpy as np
import numpy.typing as npt
from .utils.fastio import read
# from functools import singledispatch
from typing import Union

sys.modules['demucs'] = demucs  # creates a packageA entry in sys.modules

uvr_path = Path(__file__).parent
models_json_path = os.path.join(uvr_path, "models_dir", "models.json")
with open(models_json_path, "r") as f:
    models_json = json.load(f)
    
class BaseModel:
    def __init__(self, name:str, architecture:str, other_metadata:dict, device=None, logger=None):
        """Base model class

        Args:
            name (str): Model name
            architecture (str): Model architecture
            other_metadata (dict): Other metadata for the model
            device (str, optional): device to run the model on. If None the model will run gpu if available. Defaults to None.
            logger (_type_, optional): logger. Defaults to None.
        """
        self.name = name
        self.architecture = architecture
        self.other_metadata = other_metadata
        self.logger = logger
        self.remote_model_path = models_json[architecture][name]["model_path"]
        
        if device is None:
            if torch.cuda.is_available(): device = "cuda"
            elif torch.backends.mps.is_available(): device = torch.device("mps")
            else: device = "cpu"

        self.device = device
        self.model_path = download_model(model_name=name, model_path=self.remote_model_path,
                                         model_arch=architecture, logger=logger)

    def __call__(self, audio:Union[npt.NDArray, str], sampling_rate:int=None, **kwargs)->dict:
        raise NotImplementedError

    # @singledispatch
    def predict(self, audio:npt.NDArray, sampling_rate:int, **kwargs)->dict:
        raise NotImplementedError
    
    # @predict.register
    def predict_path(self, audio:str, **kwargs)->dict:
        # audio, sampling_rate = read(audio)
        # return self.predict(audio, sampling_rate)
        raise NotImplementedError
    
    def separate(self, audio:npt.NDArray, sampling_rate:int=None)->dict:
        return self.__call__(audio, sampling_rate)

    def __repr__(self):
        return f"Architecture {self.architecture}, model {self.name}. With other_metadata {self.other_metadata}"
    
    def to(self, device:str):
        """Move the model to the device

        Args:
            device (str): device to move the model to
        """
        raise NotImplementedError

    def update_metadata(self, metadata:dict):
        """Update the model metadata

        Args:
            metadata (dict): new metadata
        """
        raise NotImplementedError

    @staticmethod
    def list_models()->list:
        """List all models

        Returns:
            dict: dictionary of all models
        """
        models_list = []
        for arch in models_json.keys():
            models_list_arch = list(models_json[arch].keys())
            models_list_arch = [f"{arch}: {model}" for model in models_list_arch]
            models_list.extend(models_list_arch)
            
        return models_list

class Demucs(BaseModel):
    
    def __init__(self, other_metadata:dict, name:str="htdemucs", device=None, logger=None):
        super().__init__(name, architecture="demucs", other_metadata=other_metadata)
        self.model_path = os.path.join(uvr_path, "models_dir", "demucs", "weights", name) 
        self.model_api = demucs_api.Separator(self.name, repo=Path(self.model_path), device=self.device, **other_metadata)
        self.sample_rate = self.model_api._samplerate
  
    def predict(self, audio:npt.NDArray, sampling_rate:int, **kwargs)->dict:
        """Separate the audio into its components

        Args:
            audio (np.array): audio data
            sampling_rate (int): sampling rate

        Returns:
            dict: separated audio
        """
        if isinstance(audio, np.ndarray): 
            audio = torch.from_numpy(audio)
        elif isinstance(audio, list): 
            audio = torch.tensor(audio, dtype=torch.float32)
        
        origin, separated = self.model_api.separate_tensor(audio, sampling_rate)
        return separated
    
    def to(self, device:str):
        self.model_api.update_parameter(device=device)
        self.model_api.model.to(device)
    
    @staticmethod
    def list_models()->dict:
        return list(models_json["demucs"].keys())
    
    def update_metadata(self, metadata:dict):
        self.model_api.update_parameter(**metadata)
        self.other_metadata.update(metadata)

    def predict_path(self, audio: str, **kwargs) -> dict:
        audio, sampling_rate = read(audio)
        audio = torch.tensor(audio, dtype=torch.float32)
        return self.predict(audio, sampling_rate)
    
    def __call__(self, audio:Union[npt.NDArray, str], sampling_rate:int=None, **kwargs)->dict:
        if isinstance(audio, str):
            return self.predict_path(audio)
        return self.predict(audio, sampling_rate)

class VrNetwork(BaseModel):
    def __init__(self, other_metadata:dict, name:str="1_HP-UVR", device=None, logger=None):
        """
        Args:
            other_metadata (dict): Other metadata for the model. Most importantly the aggressiveness
            name (str, optional): Model name. Defaults to "1_HP-UVR.pth".
            device (str, optional): device to run the model on. If None the model will run gpu if available. Defaults to None.
            logger (_type_, optional): logger. Defaults to None.
        """
        super().__init__(name, architecture="vr_network", other_metadata=other_metadata)
        model_path = os.path.join(uvr_path, "models_dir", "vr_network", "weights", name) 
        if device is None:
            if torch.cuda.is_available(): device = "cuda"
            elif torch.backends.mps.is_available(): device = torch.device("mps")
            else: device = "cpu"
        
        files = os.listdir(model_path)
        for file_ in files:
            if file_.split(".")[-1] in self.allowed_model_extensions():
                self.model_path = os.path.join(model_path, file_)
                break

        model_run, mp, is_vr_51_model, stems  = vr_api.load_model(self.model_path, device)
        
        self.model_run = model_run
        self.mp = mp
        self.is_vr_51_model = is_vr_51_model
        self.stems = stems

        self.sample_rate = self.mp.param['sr']

        self.set_aggressiveness()
        self.to(device)

    def allowed_model_extensions(self):
        return ["pth", "pt", "pkl", "ckpt"]
        
    def set_aggressiveness(self):
        # make sure the aggressiveness is set
        if "aggressiveness" not in self.other_metadata: self.other_metadata["aggressiveness"] = 0.05

        self.aggressiveness = {'value': self.other_metadata["aggressiveness"], 
                                'split_bin': self.mp.param['band'][1]['crop_stop'], 
                                'aggr_correction': self.mp.param.get('aggr_correction')
                                }
    
    def set_inference_params(self, prams:Union[None, dict]):
        if prams is None: prams = {}

        if not "wav_type_set" in prams: prams["wav_type_set"] = 'PCM_U8'
        if not "window_size" in prams: prams["window_size"] = 512
        if not "post_process_threshold" in prams: prams["post_process_threshold"] = None
        if not "batch_size" in prams: prams["batch_size"] = 4
        if not "is_tta" in prams: prams["is_tta"] = False
        if not "normaliz" in prams: prams["normaliz"] = False
        if not "high_end_process" in prams: prams["high_end_process"] = False
        if not "input_high_end" in prams: prams["input_high_end"] = False
        if not "input_high_end_h" in prams: prams["input_high_end_h"] = False
        
        return prams

    def predict(self, audio:Union[npt.NDArray, str], sampling_rate:int, prams:Union[None, dict])->dict:
        
        prams = self.set_inference_params(prams)
        inp, input_high_end, input_high_end_h = vr_api.loading_mix(audio,
                                                                   self.mp, 
                                                                   self.is_vr_51_model, 
                                                                   wav_type_set=prams["wav_type_set"], 
                                                                   high_end_process=prams["high_end_process"])


        y_spec, v_spec = vr_api.inference_vr(X_spec=inp, 
                                             aggressiveness=self.aggressiveness, 
                                             window_size=prams["window_size"],
                                             model_run=self.model_run, 
                                             is_tta=prams["is_tta"], 
                                             batch_size=prams["batch_size"], 
                                             post_process_threshold=prams["post_process_threshold"], 
                                             primary_stem=self.stems["primary_stem"],
                                             device=self.device)


        audio_res = vr_api.get_audio_dict(y_spec=y_spec, 
                                          v_spec=v_spec, 
                                          stems=self.stems, 
                                          model_params=self.mp,
                                          normaliz=prams["normaliz"], 
                                          is_vr_51_model=["is_vr_51_model"],
                                          high_end_process=["high_end_process"], 
                                          input_high_end=input_high_end, 
                                          input_high_end_h=input_high_end_h)

        return audio_res
    
    def to(self, device:str):
        self.device = device
        self.model_run.to(device)

    def update_metadata(self, metadata:dict):
        self.other_metadata.update(metadata)
        self.set_aggressiveness()

    def predict_path(self, audio: str, prams:dict=None, **kwargs) -> dict:
        # audio, sampling_rate = read(audio)
        # audio = torch.tensor(audio, dtype=torch.float32)
        return self.predict(audio, None, prams)

    def __call__(self, audio:Union[npt.NDArray, str], sampling_rate:int=None, prams:dict=None, **kwargs)->dict:
        if isinstance(audio, str):
            return self.predict_path(audio, prams)
        return self.predict(audio, sampling_rate, prams)

    @staticmethod
    def list_models()->dict:
        return list(models_json["vr_network"].keys())

class MDX(BaseModel):
    models_data = mdx_api.load_mdx_models_data(model_path=os.path.join(uvr_path, "models_dir", "mdx", "modelparams", "model_data.json")) 

    def __init__(self, other_metadata:dict, name:str="UVR-MDX-NET-Inst_1", device=None, logger=None):
        super().__init__(name, architecture="mdx", other_metadata=other_metadata)
        self.sample_rate = 44100
        model_path = os.path.join(uvr_path, "models_dir", "mdx", "weights", name) 
        file_name = os.listdir(model_path)[0]
        model_path = os.path.join(model_path, file_name)
        
        model_hash = mdx_api.get_model_hash_from_path(model_path)
        model_data = MDX.models_data[model_hash]
        dim_t = model_data['mdx_dim_t_set']
        self.model_path = model_path
        self.model_hash = model_hash
        self.model_data = model_data

        if "segment_size" in other_metadata:
            segment_size = other_metadata["segment_size"]
        else:
            segment_size = 256

        model_run, (dim_c, hop) = mdx_api.load_modle(model_path, device, segment_size=segment_size, dim_t=dim_t)
        self.model_run = model_run

        other_metadata['dim_c'] = dim_c
        other_metadata['hop'] = hop
        self.device = device
        self.init_other_metadata(other_metadata)
        self.model_data_to_other_metadata(model_data)
    
    def model_data_to_other_metadata(self, model_data:dict):
        self.model_data = model_data

        other_metadata = {
            'n_fft': model_data['mdx_n_fft_scale_set'],
            'dim_f': model_data['mdx_dim_f_set'],
            'compensate': model_data['compensate'],
            'primary_stem': model_data['primary_stem'],
        }
        self.set_other_metadata(other_metadata)

    def init_other_metadata(self, other_metadata):
        self.other_metadata = {
            'segment_size': 256,
            'overlap': 0.75,
            'mdx_batch_size': 1,
            'semitone_shift': 0,
            'adjust': 1.08, 
            'denoise': False,
            'is_invert_spec': False,
            'is_match_frequency_pitch': True,
            'overlap_mdx': None
        }

        self.other_metadata.update(other_metadata)

    def set_other_metadata(self, metadata:dict):
        self.other_metadata.update(**metadata)
    
    def predict(self, audio: NDArray, sampling_rate: int, **kwargs) -> dict:
        prams = self.other_metadata

        mix = mdx_api.prepare_mix(audio)
        stems = mdx_api.demix(self.model_run, mix, prams, device=self.device)
        second_stem = mdx_api.get_secondery_stems(self.model_run, stems, mix, prams, device=self.device)
        dect_stems = mdx_api.nparray_stem_to_dict(stems, second_stem, self.model_data)

        return dect_stems

    def predict_path(self, audio: str, **kwargs) -> dict:
        audio, sampling_rate = read(audio)
        return self.predict(audio, sampling_rate)
    
    def __call__(self, audio:Union[npt.NDArray, str], sampling_rate:int=None, **kwargs)->dict:
        if isinstance(audio, str):
            return self.predict_path(audio)
        return self.predict(audio, sampling_rate)
    
    def to(self, device:str):
        self.device = device

    @staticmethod
    def list_models()->dict:
        return list(models_json["mdx"].keys())

class MDXC(BaseModel):
    models_data = mdxc_api.load_mdxc_models_data(model_path=os.path.join(uvr_path, "models_dir", "mdxc", "modelparams", "model_data.json")) 

    def __init__(self, name: str, other_metadata: dict, device=None, logger=None):
        super().__init__(name, "mdxc", other_metadata, device, logger)

        self.sample_rate = 44100
        model_path = os.path.join(uvr_path, "models_dir", "mdxc", "weights", name) 
        file_name = os.listdir(model_path)[0]
        model_path = os.path.join(model_path, file_name)
        model_hash = mdxc_api.get_model_hash_from_path(model_path=model_path)

        model_prams_dir = os.path.join(uvr_path, "models_dir", "mdxc", "modelparams") 
        model_data = mdxc_api.load_mdxc_model_data(MDXC.models_data, model_hash, model_path=model_prams_dir)
        # print(type(model_data))
        model_run = mdxc_api.load_modle(model_path, model_data, device)
        
        self.model_data = model_data
        self.model_run = model_run

        self.init_metadata()
        self.update_metadata(other_metadata)

    def init_metadata(self):
        prams = {
            'is_mdx_c_seg_def': False,
            'segment_size': 256,
            'batch_size': 1,
            'overlap_mdx23': 8,
            'semitone_shift': 0,
        }
        self.other_metadata = prams

    def update_metadata(self, other_metadata):
        self.other_metadata.update(other_metadata)
    
    def to(self, device:str):
        self.device = device
    
    def predict(self, audio: NDArray, sampling_rate: int, **kwargs) -> dict:
        mix = mdxc_api.prepare_mix(audio)
        stems = mdxc_api.demix(mix, self.other_metadata, self.model_run, self.model_data, self.device)
        stems = mdxc_api.rename_stems(stems)

        return stems
    
    def __call__(self, audio:Union[npt.NDArray, str], sampling_rate:int=None, **kwargs)->dict:
        if isinstance(audio, str):
            return self.predict_path(audio)
        return self.predict(audio, sampling_rate) 

    def predict_path(self, audio: str, **kwargs) -> dict:
        audio, sampling_rate = read(audio)
        return self.predict(audio, sampling_rate)

    @staticmethod
    def list_models()->dict:
        return list(models_json["mdxc"].keys())
    

