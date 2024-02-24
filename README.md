![Static Badge](https://img.shields.io/badge/passing-tests-blue)
![Static Badge](https://img.shields.io/badge/pre_release-red)
<a href="https://www.buymeacoffee.com/mohannadbarakat" target="_blank"><img src="https://img.shields.io/badge/-buy_me_a%C2%A0coffee-gray?logo=buy-me-a-coffee" alt="Buy Me A Coffee"></a>
<a href="https://colab.research.google.com/drive/1qf17AV5KU_8v0f29zUnPHQBbr3iX8bu6?usp=sharing" target="_blank"><img src="https://img.shields.io/badge/colab-notebook-yellow" alt="Buy Me A Coffee"></a>

# Ultimate Vocal Remover API v0.1

This is a an API for ultimate vocal removing. It is designed to be expandable with new models/algorethems while maintaining a simple interface. 
[Colab demo](https://colab.research.google.com/drive/1qf17AV5KU_8v0f29zUnPHQBbr3iX8bu6?usp=sharing)


# Install
If you intend to edit the code
```bash
git clone https://github.com/NextAudioGen/ultimatevocalremover_api.git
cd ultimatevocalremover_api
pip install .
```
# Usage
```python
import uvr
from uvr import models
from uvr.utils.get_models import download_all_models
import torch
import audiofile
import json

models_json = json.load(open("/content/ultimatevocalremover_api/src/models_dir/models.json", "r"))
download_all_models(models_json)
name = {name_of_your_audio}
device = "cuda"
    
demucs = models.Demucs(name="hdemucs_mmi", other_metadata={"segment":2, "split":True}, device=device, logger=None)

# Separating an audio file
res = demucs(name)
seperted_audio = res["separated"]
vocals = seperted_audio["vocals"]
base = seperted_audio["bass"]
drums = seperted_audio["drums"]
other = seperted_audio["other"]
```
# Archetecture:
```text
Ultimate Vocal Remover API
├── src
│   ├── audiotools.py 
│   ├── models.py 
│   ├── ensembles.py
│   ├── pipelines.py
│   ├── utils/
│   ├── audio_tools/
│   └── models_dir
│       ├── Each implementation of a model is added here as a single directory.
│       └── models.json (this is used to download the models)
├── docs
│   ├── models/
│   │   └── Here goes all models docs each in a single directory.
│   ├── ensembles/
│   │   └── Here goes all ensembles docs each in a single directory.
│   ├── pipelines/
│   │   └── Here goes all pipelines docs each in a single directory.
│   ├── audio_tools/
│   └── utils/
└── tests/
    ├── test_models.py
    ├── test_ensembles.py
    ├── test_pipelines.py
    ├── test_audiotools.py
    └── utils/
```
**audiotools.py:** Interface for all audio tools \
**models.py:** Interface for all models following a consistent interface \
**utils/** Here goes read and write utils for audio, models...etc. \

## All models, pipelines and ensembles follow this interface:
```python
class BaseModel:
    def __init__(self, name:str, architecture:str, other_metadata:dict, device=None, logger=None)
    def __call__(self, audio:Union[npt.NDArray, str], sampling_rate:int=None, **kwargs)->dict
    # @singledispatch
    def predict(self, audio:npt.NDArray, sampling_rate:int, **kwargs)->dict
    def predict_path(self, audio:str, **kwargs)->dict
    def separate(self, audio:npt.NDArray, sampling_rate:int=None)->dict
    def __repr__(self)
    def to(self, device:str)
    def update_metadata(self, metadata:dict)
    @staticmethod
    def list_models()->list

```

# Contribution
If you like this, leave a star, fork it, and definitely you are welcomed to [buy me a coffee](https://www.buymeacoffee.com/mohannadbarakat).

Also, please open issues, make pull requests but remember to follow the structure and interfaces. Moreover, we are trying to build automated testing, we are aware that the current tests are so naive but we are working on it. So please make sure to add some tests to your new code as well.

# Refrences
## code
Code and weights from these sources used in developing this library:
- [MDX-Net](https://github.com/kuielab/mdx-net/tree/main) This is the original MDX architecture implementation. 
- [MDXC and demucs](https://github.com/ZFTurbo/MVSEP-MDX23-music-separation-model/tree/main) This repo has a clever ensumbling methods for MDX, Demucs 3, and Demucs 4. Moreover they have the wieghts for their finetuned MDX open (available under MDXC implementation [here](/src/models_dir/mdxc/)).
- [Demucs](https://github.com/facebookresearch/demucs/tree/e976d93ecc3865e5757426930257e200846a520a) This is the original implementation of the model.
- [ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui/tree/master) This is one of the best vocal removers. A lot of ideas in this repo were borrowed from here.
- [weights](https://github.com/TRvlvr/model_repo/releases/tag/all_public_uvr_models) Most of the models right now are comming from this repo.

## Papers
- [Benchmarks and leaderboards for sound demixing
tasks](https://arxiv.org/pdf/2305.07489.pdf)
- [MULTI-SCALE MULTI-BAND DENSENETS FOR AUDIO SOURCE SEPARATION](https://arxiv.org/pdf/1706.09588.pdf)
- [HYBRID TRANSFORMERS FOR MUSIC SOURCE SEPARATION](https://arxiv.org/pdf/2211.08553.pdf)
- [KUIELab-MDX-Net: A Two-Stream Neural Network for Music Demixing](https://arxiv.org/abs/2111.12203)

# Core Developers

- [Mohannad Barakat](https://github.com/mohannadEhabBarakat/)
- [Noha Magdy](https://github.com/Noha-Magdy)
- [Mohtady Ehab](https://github.com/Mohtady-Ehab)
