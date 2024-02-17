import os
import glob 
import urllib.request 
from typing import List

def download_model(model_name:str, model_path:List[str], model_arch:str, logger=None)->str:
    """Download model from Hugging Face model hub
    
    Args:
        model_name (str): model name. 
        model_path (list[str]): model pathS to download the model from.
        model_arch (str): model architecture. A path in ../models_dir/{model_arch}/weights/{model_name}
                            If path is not found it will be created. And if the model is already downloaded it will not be downloaded again.
        logger (logging.Logger, optional): logger. Defaults to None.
    
    Returns:
        str: path to the downloaded model
    """
    current_path = os.getcwd()
    save_path = os.path.join(current_path, "src", "models_dir", model_arch, "weights", model_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    files = [path.split("/")[-1] for path in model_path]
    if model_exists(model_name=model_name, model_arch=model_arch, files=files):
        if logger:
            logger.info(f"Model {model_name} is already exists in {save_path}")
        return save_path
    
    try:
        # os.system(f"wget {model_path} -P {local_model_path}")
        for file_name, path in zip(files, model_path):
            local_file_path = os.path.join(save_path, file_name)
            urllib.request.urlretrieve(path, local_file_path)
            if logger:
                logger.info(f"Downloaded {model_name} from {model_path}")
            
        
        return save_path
    
    except Exception as e:
        if logger:
            logger.error(f"Failed to download {model_name} from {model_path} with error {e}")
    
    return None

def model_exists(model_name:str, model_arch:str, files:List=None)->bool:
    """Check if the model exists in ../models_dir/{model_arch}/weights/{model_name}
    
    Args:
        model_name (str): model name.
        model_arch (str): model architecture.
        files (list[str], optional): list of files to check if they exist. Defaults to None. If not provided it will check if the model directory exists.
    
    Returns:
        bool: True if the model exists, False otherwise
    """
    # remove extension from the model name
    if len(model_name.split('.')) > 1:
        model_name = model_name.split('.')[0]
    
    current_path = os.getcwd()
    save_path = os.path.join(current_path, "src", "models_dir", model_arch, "weights", model_name)
    if files is not None:
        for file in files:
            local_model_path = os.path.join(save_path, file)
            if not os.path.isfile(local_model_path):
                return False

    if os.path.exists(save_path):
        return True
    return False

"""
Example of the model json file:
models_json = {

"demucs":{
    "name1":{
        "model_path":"https://abc/bcd/model.pt",
        "other_metadata":1,
    },
    }
}
"""

def download_all_models(models_json:dict, logger=None)->dict:
    """Download all models from the models_json
    
    Args:
        models_json (dict): dictionary of models to download
        logger (logging.Logger, optional): logger. Defaults to None.

    Returns:
        dict: dictionary of downloaded models. with the same structure as the input models_json.
                architectures -> model_name -> model_path. Also the model_path will be the local path to the downloaded model.
                If the model is already downloaded it will not be downloaded again. And if the model failed to download it will be None.
    """
    paths = {}
    for model_arch, models in models_json.items():
        paths[model_arch] = {}
        for model_name, model_data in models.items():
            model_path = model_data["model_path"]
            model_path = download_model(model_name=model_name, model_path=model_path, model_arch=model_arch, logger=logger)
            paths[model_arch][model_name] = model_path

    return paths

