from ..src import models
from ..src.utils import get_models, fastio
import torch
import audiofile
from pathlib import Path
import os, json
import time
import pytest

uvr_path = os.path.join(Path(__file__).parent.parent, "src")


models_json = json.load(open(os.path.join(uvr_path, "models_dir", "models.json"), "r"))

# models_json = {"demucs" : models_json["demucs"]}

if not os.path.exists("temp.mp3"):
    os.system("wget https://huggingface.co/datasets/Mohannad/test_NextAudioGen_uvr/resolve/main/temp.mp3")

if not os.path.exists("temp.wav"):
    os.system("wget https://huggingface.co/datasets/Mohannad/test_NextAudioGen_uvr/resolve/main/temp.wav")

if not os.path.exists("temp.flac"):
    os.system("wget https://huggingface.co/datasets/Mohannad/test_NextAudioGen_uvr/resolve/main/temp.flac")

audio = "./temp"
modes = [None, "wav", "mp3", "flac"]

skip_arch = []#["vr_network"]

for arch in skip_arch:
    if arch in models_json: 
        del models_json[arch]

def get_tests():
    """
    Returns a list of tests to run
    """
    tests = []
    for arch in models_json.keys():
        for model_name in models_json[arch]:
            for mode in modes:
                if mode is None:
                    audio_ = fastio.read(audio+".mp3")
                    tests.append((arch, model_name, audio_, mode))
                else:
                    tests.append((arch, model_name, audio, mode))
    return tests

class TestModels:

    models_json = models_json
    audio = audio
    modes = modes
    models_status = {}

    @pytest.mark.parametrize('architecture, model_name, file_name, file_mode', get_tests())
    def test_a_model(self, architecture, model_name, file_name, file_mode, hyperparameters=None):

        """
        Args:
        architecture: str, the architecture of the model
        model_name: str, the name of the model
        file_name: str, the name of the audio file
        file_mode: str, the mode of the audio file. Default is "wav" so the file name should be "file_name.wav". If None then the file name should be (AudioFile, SampleRate)
        hyperparameters: dict, the hyperparameters of the model. Default is None


        each point in this list is a function that does the following:
        ✅1- download  model function and asssert that the model is downloaded. Takes model name and architecture
        ✅2- load model function and assert that the model is loaded. Takes model name, architecture and hyperparameters
        ✅3- test the model by passing an audio file and assert that the model is working. Assert input len == output len (in time not samples)
        4- del model and assert that the model is deleted (future)
        5- accumulate the time, status of each model and save it in a file
        """
        models_status = self.models_status
        passed = True
        arch = architecture
        if file_mode is not None:
            audio = file_name+"."+file_mode
        else:
            audio = file_name

        if arch not in self.models_status:
            models_status[arch] = {}
        
        if file_mode is None:
            file_mode = "audio"

        if model_name not in self.models_status[arch]:
            models_status[arch][model_name] = {}

        models_status[arch][model_name][file_mode]={
                "download": False,
                "load": False,
                "run": False,
                "time": 0
                }

        t1 = time.time()
        try:
            model_path = self.download_model(model_name, arch)
            models_status[arch][model_name][file_mode]["download"] = True
        except Exception as e:
            print(f"Failed to download {model_name} from {self.models_json[arch][model_name]['model_path']} with error {e}")
            passed = False
            
        
        try:
            model = self.load_model(model_name, arch)
            models_status[arch][model_name][file_mode]["load"] = True
        except Exception as e:
            print(f"Failed to load {model_name} from {model_path} with error {e}")
            passed = False
            
        
        try:
            seperted_audio = self.run_model(model, audio)
            models_status[arch][model_name][file_mode]["run"] = True
        except Exception as e:
            print(f"Failed to run {model_name} from {model_path} with error {e}")
            passed = False
            raise e
            

        t2 = time.time()
        models_status[arch][model_name][file_mode]["time"] = t2 - t1

        del model
        # assert not os.path.exists(model_path)

        self.models_status = models_status
        with open(os.path.join(uvr_path, "..", "tests", "models_status.json"), "w") as f:
            json.dump(models_status, f, indent=4)

        assert passed

    def download_model(self, model_name, architecture):

        model_path = get_models.download_model(model_name=model_name, model_arch=architecture)
        assert model_path is not None
        assert os.path.exists(model_path)
        assert os.path.isdir(model_path)
        # assert model_path is not empty
        assert len(os.listdir(model_path)) > 0
        return model_path
    
    def load_model(self, model_name, architecture, hyperparameters=None, logger=None):
        if torch.cuda.is_available():
            device = "cuda"  
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = "cpu"

        if architecture == "demucs":
            if hyperparameters is None:
                hyperparameters = {"segment":2, "split":True}
            model = models.Demucs(name=model_name, other_metadata=hyperparameters, device=device, logger=logger)
        elif architecture == "vr_network":
            if hyperparameters is None:
                hyperparameters = {}
            model = models.VrNetwork(name=model_name, other_metadata=hyperparameters, device=device, logger=logger)
        elif architecture == "mdx":
            if hyperparameters is None:
                hyperparameters = {}
            model = models.MDX(name=model_name, other_metadata=hyperparameters, device=device, logger=logger)
        elif architecture == "mdxc":
            if hyperparameters is None:
                hyperparameters = {}
            model = models.MDXC(name=model_name, other_metadata=hyperparameters, device=device, logger=logger)
        
        assert model is not None
        return model

    def run_model(self, model, audio_path):
        if type(audio_path) == str:
            seperted_audio = model(audio_path)
            audio, sr = audiofile.read(audio_path)
        else:
            audio, sr = audio_path
            seperted_audio = model(audio, sr)
        assert seperted_audio is not None
        for key in seperted_audio.keys():
            assert seperted_audio[key].shape[0] > 0
            assert abs(seperted_audio[key].shape[0]/model.sample_rate - audio.shape[0]/sr) < 0.1
        return seperted_audio



# def test_demucs_load():
#     if torch.cuda.is_available(): device = "cuda"
#     elif torch.backends.mps.is_available(): device = torch.device("mps")
#     else: device = "cpu"
#     print("device:", device)
#     demucs = models.Demucs(name="hdemucs_mmi", other_metadata={"segment":2, "split":True}, 
#                            device=device, logger=None)
    
#     name = "/Users/mohannadbarakat/Downloads/t.wav"
#     # Separating an audio file
#     res = demucs(name)
#     assert res is not None
#     seperted_audio = res["separated"]
#     vocals = seperted_audio["vocals"]
#     base = seperted_audio["bass"]
#     drums = seperted_audio["drums"]
#     other = seperted_audio["other"]

#     # save the separated audio
#     vocals_path = "vocals.mp3"
#     base_path = "base.mp3"
#     drums_path = "drums.mp3"
#     other_path = "other.mp3"

#     audiofile.write(vocals_path, vocals, demucs.sample_rate)
#     audiofile.write(base_path, base, demucs.sample_rate)
#     audiofile.write(drums_path, drums, demucs.sample_rate)
#     audiofile.write(other_path, other, demucs.sample_rate)

def test_vr_load():
    if torch.cuda.is_available(): device = "cuda"
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = "cpu"
    print("device:", device)
    VrNetwork = models.VrNetwork(name="1_HP-UVR", other_metadata={}, 
                           device=device, logger=None)
    
    name = "temp.wav"
    # Separating an audio file
    seperted_audio = VrNetwork(name)
    assert seperted_audio is not None
    
    print("seperted_audio:", seperted_audio.keys())

    vocals = seperted_audio["vocals"]
    instrumental = seperted_audio["instrumental"]
    vocals_path = "vocals_vr.mp3"
    instrumental_path = "instrumental_vr.mp3"
    audiofile.write(vocals_path, vocals, VrNetwork.sample_rate)
    audiofile.write(instrumental_path, instrumental, VrNetwork.sample_rate)

# def test_mdx_load():
#     if torch.cuda.is_available(): device = "cuda"
#     elif torch.backends.mps.is_available(): device = torch.device("mps")
#     else: device = "cpu"
#     print("device:", device)
#     mdx = models.MDX(name="UVR-MDX-NET-Inst_1", other_metadata={}, 
#                            device=device, logger=None)
    
#     name = "/Users/mohannadbarakat/Downloads/t.wav"
#     # Separating an audio file
#     seperted_audio = mdx(name)
#     assert seperted_audio is not None
    
#     print("seperted_audio:", seperted_audio.keys())

#     vocals = seperted_audio["vocals"]
#     instrumental = seperted_audio["instrumental"]
#     vocals_path = "vocals_mdx.mp3"
#     instrumental_path = "instrumental_mdx.mp3"
#     audiofile.write(vocals_path, vocals, mdx.sample_rate)
#     audiofile.write(instrumental_path, instrumental, mdx.sample_rate)

# def test_mdxc_load():
#     if torch.cuda.is_available(): device = "cuda"
#     elif torch.backends.mps.is_available(): device = torch.device("mps")
#     else: device = "cpu"
#     print("device:", device)
#     mdxc = models.MDXC(name="MDX23C-8KFFT-InstVoc_HQ", other_metadata={}, 
#                            device=device, logger=None)
    
#     name = "/Users/mohannadbarakat/Downloads/t.wav"
#     # Separating an audio file
#     seperted_audio = mdxc(name)
#     assert seperted_audio is not None
    
#     print("seperted_audio:", seperted_audio.keys())

#     vocals = seperted_audio["vocals"]
#     instrumental = seperted_audio["instrumental"]
#     vocals_path = "vocals_mdxc.mp3"
#     instrumental_path = "instrumental_mdxc.mp3"
#     audiofile.write(vocals_path, vocals, mdxc.sample_rate)
#     audiofile.write(instrumental_path, instrumental, mdxc.sample_rate)

# # if __name__ == "__main__":
# #     # test_demucs_load()
# #     test_vr_load()