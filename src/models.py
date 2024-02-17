import types
from .utils.get_models import download_model, model_exists
import json 
from .models_dir.demucs.demucs import api as demucs_api
from .models_dir.demucs import demucs 
import torch
import os
from pathlib import Path
import sys
import numpy.typing as npt
from .utils.fastio import read
# from functools import singledispatch
from typing import Union

sys.modules['demucs'] = demucs  # creates a packageA entry in sys.modules

current_path = os.getcwd()
models_json_path = os.path.join(current_path, "src", "models_dir", "models.json")
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

    def __call__(self, audio:Union[npt.NDArray, str], sampling_rate:int=None)->dict:
        if isinstance(audio, str):
            return self.predict_path(audio)
        return self.predict(audio, sampling_rate)

    # @singledispatch
    def predict(self, audio:npt.NDArray, sampling_rate:int)->dict:
        raise NotImplementedError
    
    # @predict.register
    def predict_path(self, audio:str)->dict:
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
        current_path = os.getcwd()
        self.model_path = os.path.join(current_path, "src", "models_dir", "demucs", "weights", name) 
        self.model_api = demucs_api.Separator(self.name, repo=Path(self.model_path), device=self.device, **other_metadata)
        self.sample_rate = self.model_api._samplerate
  
    def predict(self, audio:npt.NDArray, sampling_rate:int)->dict:
        """Separate the audio into its components

        Args:
            audio (np.array): audio data
            sampling_rate (int): sampling rate

        Returns:
            dict: separated audio
        """
        origin, separated = self.model_api.separate_tensor(audio, sampling_rate)
        return {"origin":origin, "separated":separated}
    
    def to(self, device:str):
        self.model_api.update_parameter(device=device)
        self.model_api.model.to(device)
    
    @staticmethod
    def list_models()->dict:
        return list(models_json["demucs"].keys())
    
    def update_metadata(self, metadata:dict):
        self.model_api.update_parameter(**metadata)
        self.other_metadata.update(metadata)

    def predict_path(self, audio: str) -> dict:
        audio, sampling_rate = read(audio)
        audio = torch.tensor(audio, dtype=torch.float32)
        return self.predict(audio, sampling_rate)