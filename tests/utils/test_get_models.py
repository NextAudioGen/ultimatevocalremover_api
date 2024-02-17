import os
import shutil
from ...src.utils import get_models

def is_samepath(path1, path2):
    return os.path.abspath(path1) == os.path.abspath(path2)
                                                     
def rm_models_dir(model_arch):
    current_path = os.getcwd()
    rm_path = os.path.join(current_path, "src", "models_dir", model_arch)
    rm_path = os.path.abspath(rm_path)
    # print("rm_path", rm_path)
    # os.remove(rm_path)
    shutil.rmtree(rm_path)

def test_model_dont_exists():
    model_name = "model_name"
    model_arch = "model_arch"
    assert get_models.model_exists(model_name=model_name, model_arch=model_arch) == False

def test_model_exists():
    model_name = "model_name"
    model_arch = "model_arch"
    files = ["model_name.txt"]
    current_path = os.getcwd()
    save_path = os.path.join(current_path, "src", "models_dir", model_arch, "weights", model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for file_ in files:
        local_model_path = os.path.join(save_path, file_)
        local_model_path = os.path.abspath(local_model_path)

        with open(local_model_path, 'w') as f:
            f.write("test")

    assert get_models.model_exists(model_name=model_name, model_arch=model_arch, files=files) == True
    rm_models_dir(model_arch)

def test_download_model():
    model_arch = "model_arch"
    model_name = "model_name"
    model_path = ["https://www.google.com"]
    model_file = model_path[0].split("/")[-1]

    path = get_models.download_model(model_name=model_name, model_path=model_path, model_arch=model_arch)
    current_path = os.getcwd()
    save_path = os.path.join(current_path, "src", "models_dir", model_arch, "weights", model_name)
    local_file_path = os.path.join(save_path, model_file)
    assert is_samepath(path, save_path) == True
    assert os.path.isfile(local_file_path) == True
    rm_models_dir(model_arch)

test_models_json = {
    "arch1":{
        "model1":{
            "model_path":["https://www.google.com"]
        }
    },
    "arch2":{
        "model2":{
            "model_path":["https://www.apple.com"]
        }
    }
}

def test_get_all_models():
    test_models_json_res = {
    "arch1":{
        "model1": "www.google.com"
    },
    "arch2":{
        "model2": "www.apple.com"
        }
    }

    models = get_models.download_all_models(test_models_json)
    for arch in test_models_json_res:
        assert arch in models
        for model in test_models_json_res[arch]:
            assert model in models[arch]
            # print(models[arch][model])
            current_path = os.getcwd()
            # ref_model_path = test_models_json_res[arch][model]
            ref_model_path = os.path.join(current_path, "src", "models_dir", arch, "weights", model)
            assert is_samepath(models[arch][model], ref_model_path) == True
            assert get_models.model_exists(model_name=model, model_arch=arch) == True
        
        rm_models_dir(arch)



