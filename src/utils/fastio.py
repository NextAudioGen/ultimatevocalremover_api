import numpy as np
import soundfile as sf
import librosa
import audiofile as af
import types
from typing import Union, List, Tuple
import numpy.typing as npt

def read(path:str, insure_2d:bool=True, target_sampling_rate=None, logger=None)->Tuple[npt.NDArray, int]:
    """Read audio file first try with audiofile then with soundfile and last with librosa
    
    Args:
        path (str): path to the audio file
        insure_2d (bool, optional): insure that the audio data is 2D. 
                                    If audio is dosen't have 2 channels it will be converted to 2D by repeating the channel. 
                                    Defaults to True.
        logger (logging.Logger, optional): logger. Defaults to None.

    Returns:
        tuple: audio data and samplerate

    Raises:
        ValueError: Failed to read the audio file with any of the available libraries
    
    """

    ext = path.split('.')[-1]
    signal, sampling_rate = None, None

    if ext in ['wav', 'flac', 'ogg', 'mp3']:
        try: 
            signal, sampling_rate = af.read(path)
        except Exception as e:
            if logger:
                logger.warning(f"audiofile failed to read {path} with error {e}")

    if signal is None:
        try:
            signal, sampling_rate = sf.read(path)
        except Exception as e:
            if logger:
                logger.warning(f"soundfile failed to read {path} with error {e}")

    if signal is None: 
        try:
            signal, sampling_rate = librosa.load(path, sr=None, mono=False)
        except Exception as e:
            if logger:
                logger.error(f"librosa failed to read {path} with error {e}")
        
    if target_sampling_rate is not None:
        signal = librosa.resample(signal, sampling_rate, target_sampling_rate)
        sampling_rate = target_sampling_rate

    if signal is not None:
        signal = insure_2d_signal(signal, insure_2d, logger)
        return signal, sampling_rate

    raise ValueError(f"Failed to read {path} with any of the available libraries")
    

def insure_2d_signal(signal:npt.NDArray, insure_2d:bool, logger=None)->npt.NDArray:
    """Insure that the audio data is 2D. 
    If audio is dosen't have 2 channels it will be converted to 2D by repeating the channel. 
    If audio has more than 2 channels the extra channels will be removed.
    
    Args:
        signal (np.array): audio data
        insure_2d (bool): insure that the audio data is 2D. 
        logger (logging.Logger, optional): logger. Defaults to None.

    Returns:
        np.array: 2D audio data
    
    """
    if insure_2d and signal.ndim == 1:
        signal = np.stack([signal, signal])
        if logger:
            logger.warning(f"Insured 2D signal for audio data. Original shape was {signal.shape}")
    elif insure_2d and signal.ndim > 2:
        if logger:
            logger.warning(f"Insured 2D signal for audio data. Original shape was {signal.shape}")
        signal = signal[:2]
    return signal


def write(path:str, signal:Union[npt.NDArray, List], sampling_rate:int, ext:str=None, logger=None):
    """Write audio file first try with audiofile then with soundfile and last with librosa
    
    Args:
        path (str): path to the audio file
        signal (np.array|list): audio data
        sampling_rate (int): samplerate
        ext (str, optional): file extension ovverides the file extension from the path. Defaults to None. Example: 'wav', 'flac', 'ogg', 'mp3' don't add the dot.
        logger (logging.Logger, optional): logger. Defaults to None.
    
    Raises:
        ValueError: Failed to write the audio file with any of the available libraries
    """
    if ext is not None:
        path = path+'.'+ext


    if ext in ['wav', 'flac', 'ogg', 'mp3']:
        try: 
            af.write(path, signal, sampling_rate)
            return
        except Exception as e:
            if logger:
                logger.warning(f"audiofile failed to write {path} with error {e}")

    try:
        sf.write(path, signal.T, sampling_rate)
        return
    except Exception as e:
        if logger:
            logger.warning(f"soundfile failed to write {path} with error {e}")

    try:
        librosa.output.write_wav(path, signal.T, sampling_rate)
        return
    except Exception as e:
        if logger:
            logger.error(f"librosa failed to write {path} with error {e}")
    
    raise ValueError(f"Failed to write {path} with any of the available libraries")

