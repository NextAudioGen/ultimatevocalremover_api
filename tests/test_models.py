from ..src import models
import torch
import audiofile



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


def test_vr_load():
    if torch.cuda.is_available(): device = "cuda"
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = "cpu"
    print("device:", device)
    VrNetwork = models.VrNetwork(name="1_HP-UVR", other_metadata={}, 
                           device=device, logger=None)
    
    name = "/Users/mohannadbarakat/Downloads/t.wav"
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



if __name__ == "__main__":
    # test_demucs_load()
    test_vr_load()