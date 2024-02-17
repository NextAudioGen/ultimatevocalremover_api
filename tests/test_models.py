from ..src import models
import torch

def test_demucs():
    if torch.cuda.is_available(): device = "cuda"
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = "cpu"

    demucs = models.Demucs(name="hdemucs_mmi", architecture="demucs",
                            other_metadata={}, device=device,
                            logger=None)
    
    assert demucs.architecture == "demucs"