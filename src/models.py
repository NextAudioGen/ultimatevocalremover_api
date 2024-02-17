import types
from .utils.get_models import download_model, model_exists
import json 
from .models_dir.demucs.demucs import api as demucs_api
import torch
import os

with open("models.json", "r") as f:
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
                                         architecture=architecture, logger=logger)

    def __call__(self, audio, sampling_rate)->dict:
        raise NotImplementedError

    def predict(self, audio)->dict:
        return self.__call__(audio)
    
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
        super().__init__(name, architecture="Demucs", other_metadata=other_metadata)
        current_path = os.getcwd()
        self.model_path = os.path.join(current_path, "src", "models_dir", "demucs", "weights", name) 
        self.model_api = demucs_api.Separator(self.name, repo=self.model_path, device=self.device, **other_metadata)

    def __call__(self, audio, sampling_rate)->dict:
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