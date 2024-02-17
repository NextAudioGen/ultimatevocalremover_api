import os
import glob 

def download_model(model_name:str, model_path:str, model_arch:str, logger=None)->str:
    """Download model from Hugging Face model hub
    
    Args:
        model_name (str): model name.
        model_path (str): model path to download the model from.
        model_arch (str): model architecture. A path in ../models_dir/{model_arch}/weights/{model_name}
                            If path is not found it will be created. And if the model is already downloaded it will not be downloaded again.
        logger (logging.Logger, optional): logger. Defaults to None.
    
    Returns:
        str: path to the downloaded model
    """
    current_path = os.getcwd()
    save_path = os.path.join(current_path, "..", "models_dir", model_arch, "weights")
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    local_model_path = os.path.join(save_path, model_name)
    if os.path.isfile(model_path):
        if logger:
            logger.info(f"Model {model_name} is already exists in {local_model_path}")
        return local_model_path
    
    try:
        os.system(f"wget {model_path} -P {local_model_path}")
        if logger:
            logger.info(f"Downloaded {model_name} from {model_path}")
        
        
        return model_path
    
    except Exception as e:
        if logger:
            logger.error(f"Failed to download {model_name} from {model_path} with error {e}")
    
    return None

def model_exists(model_name:str, model_arch:str)->bool:
    """Check if the model exists
    
    Args:
        model_name (str): model name.
        model_arch (str): model architecture. A path in ../models_dir/{model_arch}/weights/{model_name}
    
    Returns:
        bool: True if the model exists, False otherwise
    """
    # remove extension from the model name
    if len(model_name.split('.')) > 1:
        model_name = model_name.split('.')[0]
    
    current_path = os.getcwd()
    save_path = os.path.join(current_path, "..", "models_dir", model_arch, "weights")
    local_model_path = os.path.join(save_path, model_name)
    local_model_path += ".*" # check for any file with the same name
    for name in glob.glob(local_model_path): 
        if os.path.isfile(name):
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
            model_path = download_model(model_name, model_path, model_arch, logger=logger)
            paths[model_arch][model_name] = model_path

    return paths

