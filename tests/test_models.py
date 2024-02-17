from ..src import models
from ..src.models_dir.demucs import demucs as demucs
import torch



def test_demucs_load():
    if torch.cuda.is_available(): device = "cuda"
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = "cpu"

    demucs = models.Demucs(name="hdemucs_mmi", other_metadata={}, 
                           device=device, logger=None)
    
    assert demucs.architecture == "demucs"


def test_demucs_load():
    if torch.cuda.is_available(): device = "cuda"
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = "cpu"

    demucs = models.Demucs(name="hdemucs_mmi", other_metadata={}, 
                           device=device, logger=None)
    
    name = ""
    # Separating an audio file
    res = demucs(name)
    assert res is not None