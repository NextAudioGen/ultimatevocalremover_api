from ..src import models
import torch
import audiofile



# def test_demucs_load():
#     if torch.cuda.is_available(): device = "cuda"
#     elif torch.backends.mps.is_available(): device = torch.device("mps")
#     else: device = "cpu"

#     demucs = models.Demucs(name="hdemucs_mmi", other_metadata={}, 
#                            device=device, logger=None)
    
#     assert demucs.architecture == "demucs"


def test_demucs_load():
    if torch.cuda.is_available(): device = "cuda"
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = "cpu"
    print("device:", device)
    demucs = models.Demucs(name="hdemucs_mmi", other_metadata={"segment":2, "split":True}, 
                           device=device, logger=None)
    
    name = ""
    # Separating an audio file
    res = demucs(name)
    assert res is not None
    seperted_audio = res["separated"]
    vocals = seperted_audio["vocals"]
    base = seperted_audio["bass"]
    drums = seperted_audio["drums"]
    other = seperted_audio["other"]

    # save the separated audio
    vocals_path = "vocals.mp3"
    base_path = "base.mp3"
    drums_path = "drums.mp3"
    other_path = "other.mp3"

    audiofile.write(vocals_path, vocals, demucs.sample_rate)
    audiofile.write(base_path, base, demucs.sample_rate)
    audiofile.write(drums_path, drums, demucs.sample_rate)
    audiofile.write(other_path, other, demucs.sample_rate)

# if __name__ == "__main__":
#     test_demucs_load()