# GUI modules
import time
import audioread
import hashlib
import json
import librosa
import os
from pyglet import font as pyglet_font
import shutil
import soundfile as sf
import traceback
import matchering as match
import tkinter as tk

from tkinter import messagebox
from collections import Counter
from __version__ import VERSION, PATCH, PATCH_MAC, PATCH_LINUX
from gui_data.constants import *
from gui_data.app_size_values import *
from gui_data.error_handling import error_text, error_dialouge
from gui_data.old_data_check import file_check, remove_unneeded_yamls, remove_temps
from gui_data.tkinterdnd2 import TkinterDnD, DND_FILES
from lib_v5.vr_network.model_param_init import ModelParameters
from kthread import KThread
from lib_v5 import spec_utils
from pathlib  import Path
from separate import (
    SeperateDemucs, SeperateMDX, SeperateMDXC, SeperateVR,  # Model-related
    save_format, clear_gpu_cache,  # Utility functions
    cuda_available, mps_available, #directml_available,
)
from playsound import playsound
from typing import List
import re
import sys
import yaml
from ml_collections import ConfigDict
from collections import Counter

# if not is_macos:
#     import torch_directml

# is_choose_arch = cuda_available and directml_available
# is_opencl_only = not cuda_available and directml_available
# is_cuda_only = cuda_available and not directml_available

is_gpu_available = cuda_available or mps_available# or directml_available

# Change the current working directory to the directory
# this file sits in
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app
    # path into variable _MEIPASS'.
    BASE_PATH = sys._MEIPASS
else:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

os.chdir(BASE_PATH)  # Change the current working directory to the base path

SPLASH_DOC = os.path.join(BASE_PATH, 'tmp', 'splash.txt')

if os.path.isfile(SPLASH_DOC):
    os.remove(SPLASH_DOC)

def get_execution_time(function, name):
    start = time.time()
    function()
    end = time.time()
    time_difference = end - start
    print(f'{name} Execution Time: ', time_difference)

PREVIOUS_PATCH_WIN = 'UVR_Patch_10_6_23_4_27'

is_dnd_compatible = True
banner_placement = -2


debugger = []

#--Constants--
#Models
MODELS_DIR = os.path.join(BASE_PATH, 'models')
VR_MODELS_DIR = os.path.join(MODELS_DIR, 'VR_Models')
MDX_MODELS_DIR = os.path.join(MODELS_DIR, 'MDX_Net_Models')
DEMUCS_MODELS_DIR = os.path.join(MODELS_DIR, 'Demucs_Models')
DEMUCS_NEWER_REPO_DIR = os.path.join(DEMUCS_MODELS_DIR, 'v3_v4_repo')
MDX_MIXER_PATH = os.path.join(BASE_PATH, 'lib_v5', 'mixer.ckpt')

#Cache & Parameters
VR_HASH_DIR = os.path.join(VR_MODELS_DIR, 'model_data')
VR_HASH_JSON = os.path.join(VR_MODELS_DIR, 'model_data', 'model_data.json')
MDX_HASH_DIR = os.path.join(MDX_MODELS_DIR, 'model_data')
MDX_HASH_JSON = os.path.join(MDX_HASH_DIR, 'model_data.json')
MDX_C_CONFIG_PATH = os.path.join(MDX_HASH_DIR, 'mdx_c_configs')

DEMUCS_MODEL_NAME_SELECT = os.path.join(DEMUCS_MODELS_DIR, 'model_data', 'model_name_mapper.json')
MDX_MODEL_NAME_SELECT = os.path.join(MDX_MODELS_DIR, 'model_data', 'model_name_mapper.json')
ENSEMBLE_CACHE_DIR = os.path.join(BASE_PATH, 'gui_data', 'saved_ensembles')
SETTINGS_CACHE_DIR = os.path.join(BASE_PATH, 'gui_data', 'saved_settings')
VR_PARAM_DIR = os.path.join(BASE_PATH, 'lib_v5', 'vr_network', 'modelparams')
SAMPLE_CLIP_PATH = os.path.join(BASE_PATH, 'temp_sample_clips')
ENSEMBLE_TEMP_PATH = os.path.join(BASE_PATH, 'ensemble_temps')
DOWNLOAD_MODEL_CACHE = os.path.join(BASE_PATH, 'gui_data', 'model_manual_download.json')

DENOISER_MODEL_PATH = os.path.join(VR_MODELS_DIR, 'UVR-DeNoise-Lite.pth')
DEVERBER_MODEL_PATH = os.path.join(VR_MODELS_DIR, 'UVR-DeEcho-DeReverb.pth')

MODEL_DATA_URLS = [VR_MODEL_DATA_LINK, MDX_MODEL_DATA_LINK, MDX_MODEL_NAME_DATA_LINK, DEMUCS_MODEL_NAME_DATA_LINK]
MODEL_DATA_FILES = [VR_HASH_JSON, MDX_HASH_JSON, MDX_MODEL_NAME_SELECT, DEMUCS_MODEL_NAME_SELECT]

file_check(os.path.join(MODELS_DIR, 'Main_Models'), VR_MODELS_DIR)
file_check(os.path.join(DEMUCS_MODELS_DIR, 'v3_repo'), DEMUCS_NEWER_REPO_DIR)
remove_unneeded_yamls(DEMUCS_MODELS_DIR)

remove_temps(ENSEMBLE_TEMP_PATH)
remove_temps(SAMPLE_CLIP_PATH)
remove_temps(os.path.join(BASE_PATH, 'img'))

if not os.path.isdir(ENSEMBLE_TEMP_PATH):
    os.mkdir(ENSEMBLE_TEMP_PATH)
    
if not os.path.isdir(SAMPLE_CLIP_PATH):
    os.mkdir(SAMPLE_CLIP_PATH)

model_hash_table = {}



class ModelData():
    def __init__(self, model_name: str, 
                 selected_process_method=ENSEMBLE_MODE, 
                 is_secondary_model=False, 
                 primary_model_primary_stem=None, 
                 is_primary_model_primary_stem_only=False, 
                 is_primary_model_secondary_stem_only=False, 
                 is_pre_proc_model=False,
                 is_dry_check=False,
                 is_change_def=False,
                 is_get_hash_dir_only=False,
                 is_vocal_split_model=False):

        device_set = root.device_set_var.get()
        self.DENOISER_MODEL = DENOISER_MODEL_PATH
        self.DEVERBER_MODEL = DEVERBER_MODEL_PATH
        self.is_deverb_vocals = root.is_deverb_vocals_var.get() if os.path.isfile(DEVERBER_MODEL_PATH) else False
        self.deverb_vocal_opt = DEVERB_MAPPER[root.deverb_vocal_opt_var.get()]
        self.is_denoise_model = True if root.denoise_option_var.get() == DENOISE_M and os.path.isfile(DENOISER_MODEL_PATH) else False
        self.is_gpu_conversion = 0 if root.is_gpu_conversion_var.get() else -1
        self.is_normalization = root.is_normalization_var.get()#
        self.is_use_opencl = False#True if is_opencl_only else root.is_use_opencl_var.get()
        self.is_primary_stem_only = root.is_primary_stem_only_var.get()
        self.is_secondary_stem_only = root.is_secondary_stem_only_var.get()
        self.is_denoise = True if not root.denoise_option_var.get() == DENOISE_NONE else False
        self.is_mdx_c_seg_def = root.is_mdx_c_seg_def_var.get()#
        self.mdx_batch_size = 1 if root.mdx_batch_size_var.get() == DEF_OPT else int(root.mdx_batch_size_var.get())
        self.mdxnet_stem_select = root.mdxnet_stems_var.get() 
        self.overlap = float(root.overlap_var.get()) if not root.overlap_var.get() == DEFAULT else 0.25
        self.overlap_mdx = float(root.overlap_mdx_var.get()) if not root.overlap_mdx_var.get() == DEFAULT else root.overlap_mdx_var.get()
        self.overlap_mdx23 = int(float(root.overlap_mdx23_var.get()))
        self.semitone_shift = float(root.semitone_shift_var.get())
        self.is_pitch_change = False if self.semitone_shift == 0 else True
        self.is_match_frequency_pitch = root.is_match_frequency_pitch_var.get()
        self.is_mdx_ckpt = False
        self.is_mdx_c = False
        self.is_mdx_combine_stems = root.is_mdx23_combine_stems_var.get()#
        self.mdx_c_configs = None
        self.mdx_model_stems = []
        self.mdx_dim_f_set = None
        self.mdx_dim_t_set = None
        self.mdx_stem_count = 1
        self.compensate = None
        self.mdx_n_fft_scale_set = None
        self.wav_type_set = root.wav_type_set#
        self.device_set = device_set.split(':')[-1].strip() if ':' in device_set else device_set
        self.mp3_bit_set = root.mp3_bit_set_var.get()
        self.save_format = root.save_format_var.get()
        self.is_invert_spec = root.is_invert_spec_var.get()#
        self.is_mixer_mode = False#
        self.demucs_stems = root.demucs_stems_var.get()
        self.is_demucs_combine_stems = root.is_demucs_combine_stems_var.get()
        self.demucs_source_list = []
        self.demucs_stem_count = 0
        self.mixer_path = MDX_MIXER_PATH
        self.model_name = model_name
        self.process_method = selected_process_method
        self.model_status = False if self.model_name == CHOOSE_MODEL or self.model_name == NO_MODEL else True
        self.primary_stem = None
        self.secondary_stem = None
        self.primary_stem_native = None
        self.is_ensemble_mode = False
        self.ensemble_primary_stem = None
        self.ensemble_secondary_stem = None
        self.primary_model_primary_stem = primary_model_primary_stem
        self.is_secondary_model = True if is_vocal_split_model else is_secondary_model
        self.secondary_model = None
        self.secondary_model_scale = None
        self.demucs_4_stem_added_count = 0
        self.is_demucs_4_stem_secondaries = False
        self.is_4_stem_ensemble = False
        self.pre_proc_model = None
        self.pre_proc_model_activated = False
        self.is_pre_proc_model = is_pre_proc_model
        self.is_dry_check = is_dry_check
        self.model_samplerate = 44100
        self.model_capacity = 32, 128
        self.is_vr_51_model = False
        self.is_demucs_pre_proc_model_inst_mix = False
        self.manual_download_Button = None
        self.secondary_model_4_stem = []
        self.secondary_model_4_stem_scale = []
        self.secondary_model_4_stem_names = []
        self.secondary_model_4_stem_model_names_list = []
        self.all_models = []
        self.secondary_model_other = None
        self.secondary_model_scale_other = None
        self.secondary_model_bass = None
        self.secondary_model_scale_bass = None
        self.secondary_model_drums = None
        self.secondary_model_scale_drums = None
        self.is_multi_stem_ensemble = False
        self.is_karaoke = False
        self.is_bv_model = False
        self.bv_model_rebalance = 0
        self.is_sec_bv_rebalance = False
        self.is_change_def = is_change_def
        self.model_hash_dir = None
        self.is_get_hash_dir_only = is_get_hash_dir_only
        self.is_secondary_model_activated = False
        self.vocal_split_model = None
        self.is_vocal_split_model = is_vocal_split_model
        self.is_vocal_split_model_activated = False
        self.is_save_inst_vocal_splitter = root.is_save_inst_set_vocal_splitter_var.get()
        self.is_inst_only_voc_splitter = root.check_only_selection_stem(INST_STEM_ONLY)
        self.is_save_vocal_only = root.check_only_selection_stem(IS_SAVE_VOC_ONLY)

        if selected_process_method == ENSEMBLE_MODE:
            self.process_method, _, self.model_name = model_name.partition(ENSEMBLE_PARTITION)
            self.model_and_process_tag = model_name
            self.ensemble_primary_stem, self.ensemble_secondary_stem = root.return_ensemble_stems()
            
            is_not_secondary_or_pre_proc = not is_secondary_model and not is_pre_proc_model
            self.is_ensemble_mode = is_not_secondary_or_pre_proc
            
            if root.ensemble_main_stem_var.get() == FOUR_STEM_ENSEMBLE:
                self.is_4_stem_ensemble = self.is_ensemble_mode
            elif root.ensemble_main_stem_var.get() == MULTI_STEM_ENSEMBLE and root.chosen_process_method_var.get() == ENSEMBLE_MODE:
                self.is_multi_stem_ensemble = True

            is_not_vocal_stem = self.ensemble_primary_stem != VOCAL_STEM
            self.pre_proc_model_activated = root.is_demucs_pre_proc_model_activate_var.get() if is_not_vocal_stem else False

        if self.process_method == VR_ARCH_TYPE:
            self.is_secondary_model_activated = root.vr_is_secondary_model_activate_var.get() if not is_secondary_model else False
            self.aggression_setting = float(int(root.aggression_setting_var.get())/100)
            self.is_tta = root.is_tta_var.get()
            self.is_post_process = root.is_post_process_var.get()
            self.window_size = int(root.window_size_var.get())
            self.batch_size = 1 if root.batch_size_var.get() == DEF_OPT else int(root.batch_size_var.get())
            self.crop_size = int(root.crop_size_var.get())
            self.is_high_end_process = 'mirroring' if root.is_high_end_process_var.get() else 'None'
            self.post_process_threshold = float(root.post_process_threshold_var.get())
            self.model_capacity = 32, 128
            self.model_path = os.path.join(VR_MODELS_DIR, f"{self.model_name}.pth")
            self.get_model_hash()
            if self.model_hash:
                self.model_hash_dir = os.path.join(VR_HASH_DIR, f"{self.model_hash}.json")
                if is_change_def:
                    self.model_data = self.change_model_data()
                else:
                    self.model_data = self.get_model_data(VR_HASH_DIR, root.vr_hash_MAPPER) if not self.model_hash == WOOD_INST_MODEL_HASH else WOOD_INST_PARAMS
                if self.model_data:
                    vr_model_param = os.path.join(VR_PARAM_DIR, "{}.json".format(self.model_data["vr_model_param"]))
                    self.primary_stem = self.model_data["primary_stem"]
                    self.secondary_stem = secondary_stem(self.primary_stem)
                    self.vr_model_param = ModelParameters(vr_model_param)
                    self.model_samplerate = self.vr_model_param.param['sr']
                    self.primary_stem_native = self.primary_stem
                    if "nout" in self.model_data.keys() and "nout_lstm" in self.model_data.keys():
                        self.model_capacity = self.model_data["nout"], self.model_data["nout_lstm"]
                        self.is_vr_51_model = True
                    self.check_if_karaokee_model()
   
                else:
                    self.model_status = False
                
        if self.process_method == MDX_ARCH_TYPE:
            self.is_secondary_model_activated = root.mdx_is_secondary_model_activate_var.get() if not is_secondary_model else False
            self.margin = int(root.margin_var.get())
            self.chunks = 0
            self.mdx_segment_size = int(root.mdx_segment_size_var.get())
            self.get_mdx_model_path()
            self.get_model_hash()
            if self.model_hash:
                self.model_hash_dir = os.path.join(MDX_HASH_DIR, f"{self.model_hash}.json")
                if is_change_def:
                    self.model_data = self.change_model_data()
                else:
                    self.model_data = self.get_model_data(MDX_HASH_DIR, root.mdx_hash_MAPPER)
                if self.model_data:
                    
                    if "config_yaml" in self.model_data:
                        self.is_mdx_c = True
                        config_path = os.path.join(MDX_C_CONFIG_PATH, self.model_data["config_yaml"])
                        if os.path.isfile(config_path):
                            with open(config_path) as f:
                                config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

                            self.mdx_c_configs = config
                                
                            if self.mdx_c_configs.training.target_instrument:
                                # Use target_instrument as the primary stem and set 4-stem ensemble to False
                                target = self.mdx_c_configs.training.target_instrument
                                self.mdx_model_stems = [target]
                                self.primary_stem = target
                            else:
                                # If no specific target_instrument, use all instruments in the training config
                                self.mdx_model_stems = self.mdx_c_configs.training.instruments
                                self.mdx_stem_count = len(self.mdx_model_stems)
                                
                                # Set primary stem based on stem count
                                if self.mdx_stem_count == 2:
                                    self.primary_stem = self.mdx_model_stems[0]
                                else:
                                    self.primary_stem = self.mdxnet_stem_select
                                
                                # Update mdxnet_stem_select based on ensemble mode
                                if self.is_ensemble_mode:
                                    self.mdxnet_stem_select = self.ensemble_primary_stem
                        else:
                            self.model_status = False
                    else:
                        self.compensate = self.model_data["compensate"] if root.compensate_var.get() == AUTO_SELECT else float(root.compensate_var.get())
                        self.mdx_dim_f_set = self.model_data["mdx_dim_f_set"]
                        self.mdx_dim_t_set = self.model_data["mdx_dim_t_set"]
                        self.mdx_n_fft_scale_set = self.model_data["mdx_n_fft_scale_set"]
                        self.primary_stem = self.model_data["primary_stem"]
                        self.primary_stem_native = self.model_data["primary_stem"]
                        self.check_if_karaokee_model()
                        
                    self.secondary_stem = secondary_stem(self.primary_stem)
                else:
                    self.model_status = False

        if self.process_method == DEMUCS_ARCH_TYPE:
            self.is_secondary_model_activated = root.demucs_is_secondary_model_activate_var.get() if not is_secondary_model else False
            if not self.is_ensemble_mode:
                self.pre_proc_model_activated = root.is_demucs_pre_proc_model_activate_var.get() if not root.demucs_stems_var.get() in [VOCAL_STEM, INST_STEM] else False
            self.margin_demucs = int(root.margin_demucs_var.get())
            self.chunks_demucs = 0
            self.shifts = int(root.shifts_var.get())
            self.is_split_mode = root.is_split_mode_var.get()
            self.segment = root.segment_var.get()
            self.is_chunk_demucs = root.is_chunk_demucs_var.get()
            self.is_primary_stem_only = root.is_primary_stem_only_var.get() if self.is_ensemble_mode else root.is_primary_stem_only_Demucs_var.get() 
            self.is_secondary_stem_only = root.is_secondary_stem_only_var.get() if self.is_ensemble_mode else root.is_secondary_stem_only_Demucs_var.get()
            self.get_demucs_model_data()
            self.get_demucs_model_path()
            
        if self.model_status:
            self.model_basename = os.path.splitext(os.path.basename(self.model_path))[0]
        else:
            self.model_basename = None
            
        self.pre_proc_model_activated = self.pre_proc_model_activated if not self.is_secondary_model else False
        
        self.is_primary_model_primary_stem_only = is_primary_model_primary_stem_only
        self.is_primary_model_secondary_stem_only = is_primary_model_secondary_stem_only

        is_secondary_activated_and_status = self.is_secondary_model_activated and self.model_status
        is_demucs = self.process_method == DEMUCS_ARCH_TYPE
        is_all_stems = root.demucs_stems_var.get() == ALL_STEMS
        is_valid_ensemble = not self.is_ensemble_mode and is_all_stems and is_demucs
        is_multi_stem_ensemble_demucs = self.is_multi_stem_ensemble and is_demucs

        if is_secondary_activated_and_status:
            if is_valid_ensemble or self.is_4_stem_ensemble or is_multi_stem_ensemble_demucs:
                for key in DEMUCS_4_SOURCE_LIST:
                    self.secondary_model_data(key)
                    self.secondary_model_4_stem.append(self.secondary_model)
                    self.secondary_model_4_stem_scale.append(self.secondary_model_scale)
                    self.secondary_model_4_stem_names.append(key)
                
                self.demucs_4_stem_added_count = sum(i is not None for i in self.secondary_model_4_stem)
                self.is_secondary_model_activated = any(i is not None for i in self.secondary_model_4_stem)
                self.demucs_4_stem_added_count -= 1 if self.is_secondary_model_activated else 0
                
                if self.is_secondary_model_activated:
                    self.secondary_model_4_stem_model_names_list = [i.model_basename if i is not None else None for i in self.secondary_model_4_stem]
                    self.is_demucs_4_stem_secondaries = True
            else:
                primary_stem = self.ensemble_primary_stem if self.is_ensemble_mode and is_demucs else self.primary_stem
                self.secondary_model_data(primary_stem)

        if self.process_method == DEMUCS_ARCH_TYPE and not is_secondary_model:
            if self.demucs_stem_count >= 3 and self.pre_proc_model_activated:
                self.pre_proc_model = root.process_determine_demucs_pre_proc_model(self.primary_stem)
                self.pre_proc_model_activated = True if self.pre_proc_model else False
                self.is_demucs_pre_proc_model_inst_mix = root.is_demucs_pre_proc_model_inst_mix_var.get() if self.pre_proc_model else False

        if self.is_vocal_split_model and self.model_status:
            self.is_secondary_model_activated = False
            if self.is_bv_model:
                primary = BV_VOCAL_STEM if self.primary_stem_native == VOCAL_STEM else LEAD_VOCAL_STEM
            else:
                primary = LEAD_VOCAL_STEM if self.primary_stem_native == VOCAL_STEM else BV_VOCAL_STEM
            self.primary_stem, self.secondary_stem = primary, secondary_stem(primary)
            
        self.vocal_splitter_model_data()
            
    def vocal_splitter_model_data(self):
        if not self.is_secondary_model and self.model_status:
            self.vocal_split_model = root.process_determine_vocal_split_model()
            self.is_vocal_split_model_activated = True if self.vocal_split_model else False
            
            if self.vocal_split_model:
                if self.vocal_split_model.bv_model_rebalance:
                    self.is_sec_bv_rebalance = True
            
    def secondary_model_data(self, primary_stem):
        secondary_model_data = root.process_determine_secondary_model(self.process_method, primary_stem, self.is_primary_stem_only, self.is_secondary_stem_only)
        self.secondary_model = secondary_model_data[0]
        self.secondary_model_scale = secondary_model_data[1]
        self.is_secondary_model_activated = False if not self.secondary_model else True
        if self.secondary_model:
            self.is_secondary_model_activated = False if self.secondary_model.model_basename == self.model_basename else True
            
        #print("self.is_secondary_model_activated: ", self.is_secondary_model_activated)
              
    def check_if_karaokee_model(self):
        if IS_KARAOKEE in self.model_data.keys():
            self.is_karaoke = self.model_data[IS_KARAOKEE]
        if IS_BV_MODEL in self.model_data.keys():
            self.is_bv_model = self.model_data[IS_BV_MODEL]#
        if IS_BV_MODEL_REBAL in self.model_data.keys() and self.is_bv_model:
            self.bv_model_rebalance = self.model_data[IS_BV_MODEL_REBAL]#
   
    def get_mdx_model_path(self):
        
        if self.model_name.endswith(CKPT):
            self.is_mdx_ckpt = True

        ext = '' if self.is_mdx_ckpt else ONNX
        
        for file_name, chosen_mdx_model in root.mdx_name_select_MAPPER.items():
            if self.model_name in chosen_mdx_model:
                if file_name.endswith(CKPT):
                    ext = ''
                self.model_path = os.path.join(MDX_MODELS_DIR, f"{file_name}{ext}")
                break
        else:
            self.model_path = os.path.join(MDX_MODELS_DIR, f"{self.model_name}{ext}")
            
        self.mixer_path = os.path.join(MDX_MODELS_DIR, f"mixer_val.ckpt")
    
    def get_demucs_model_path(self):
        
        demucs_newer = self.demucs_version in {DEMUCS_V3, DEMUCS_V4}
        demucs_model_dir = DEMUCS_NEWER_REPO_DIR if demucs_newer else DEMUCS_MODELS_DIR
        
        for file_name, chosen_model in root.demucs_name_select_MAPPER.items():
            if self.model_name == chosen_model:
                self.model_path = os.path.join(demucs_model_dir, file_name)
                break
        else:
            self.model_path = os.path.join(DEMUCS_NEWER_REPO_DIR, f'{self.model_name}.yaml')

    def get_demucs_model_data(self):

        self.demucs_version = DEMUCS_V4

        for key, value in DEMUCS_VERSION_MAPPER.items():
            if value in self.model_name:
                self.demucs_version = key

        if DEMUCS_UVR_MODEL in self.model_name:
            self.demucs_source_list, self.demucs_source_map, self.demucs_stem_count = DEMUCS_2_SOURCE, DEMUCS_2_SOURCE_MAPPER, 2
        else:
            self.demucs_source_list, self.demucs_source_map, self.demucs_stem_count = DEMUCS_4_SOURCE, DEMUCS_4_SOURCE_MAPPER, 4

        if not self.is_ensemble_mode:
            self.primary_stem = PRIMARY_STEM if self.demucs_stems == ALL_STEMS else self.demucs_stems
            self.secondary_stem = secondary_stem(self.primary_stem)
            
    def get_model_data(self, model_hash_dir, hash_mapper:dict):
        model_settings_json = os.path.join(model_hash_dir, f"{self.model_hash}.json")

        if os.path.isfile(model_settings_json):
            with open(model_settings_json, 'r') as json_file:
                return json.load(json_file)
        else:
            for hash, settings in hash_mapper.items():
                if self.model_hash in hash:
                    return settings

            return self.get_model_data_from_popup()

    def change_model_data(self):
        if self.is_get_hash_dir_only:
            return None
        else:
            return self.get_model_data_from_popup()

    def get_model_data_from_popup(self):
        if self.is_dry_check:
            return None
            
        if not self.is_change_def:
            confirm = messagebox.askyesno(
                title=UNRECOGNIZED_MODEL[0],
                message=f'"{self.model_name}"{UNRECOGNIZED_MODEL[1]}',
                parent=root
            )
            if not confirm:
                return None
        
        if self.process_method == VR_ARCH_TYPE:
            root.pop_up_vr_param(self.model_hash)
            return root.vr_model_params
        elif self.process_method == MDX_ARCH_TYPE:
            root.pop_up_mdx_model(self.model_hash, self.model_path)
            return root.mdx_model_params

    def get_model_hash(self):
        self.model_hash = None
        
        if not os.path.isfile(self.model_path):
            self.model_status = False
            self.model_hash is None
        else:
            if model_hash_table:
                for (key, value) in model_hash_table.items():
                    if self.model_path == key:
                        self.model_hash = value
                        break
                    
            if not self.model_hash:
                try:
                    with open(self.model_path, 'rb') as f:
                        f.seek(- 10000 * 1024, 2)
                        self.model_hash = hashlib.md5(f.read()).hexdigest()
                except:
                    self.model_hash = hashlib.md5(open(self.model_path,'rb').read()).hexdigest()
                    
                table_entry = {self.model_path: self.model_hash}
                model_hash_table.update(table_entry)
                
        #print(self.model_name," - ", self.model_hash)

class Ensembler():
    def __init__(self, is_manual_ensemble=False):
        self.is_save_all_outputs_ensemble = root.is_save_all_outputs_ensemble_var.get()
        chosen_ensemble_name = '{}'.format(root.chosen_ensemble_var.get().replace(" ", "_")) if not root.chosen_ensemble_var.get() == CHOOSE_ENSEMBLE_OPTION else 'Ensembled'
        ensemble_algorithm = root.ensemble_type_var.get().partition("/")
        ensemble_main_stem_pair = root.ensemble_main_stem_var.get().partition("/")
        time_stamp = round(time.time())
        self.audio_tool = MANUAL_ENSEMBLE
        self.main_export_path = Path(root.export_path_var.get())
        self.chosen_ensemble = f"_{chosen_ensemble_name}" if root.is_append_ensemble_name_var.get() else ''
        ensemble_folder_name = self.main_export_path if self.is_save_all_outputs_ensemble else ENSEMBLE_TEMP_PATH
        self.ensemble_folder_name = os.path.join(ensemble_folder_name, '{}_Outputs_{}'.format(chosen_ensemble_name, time_stamp))
        self.is_testing_audio = f"{time_stamp}_" if root.is_testing_audio_var.get() else ''
        self.primary_algorithm = ensemble_algorithm[0]
        self.secondary_algorithm = ensemble_algorithm[2]
        self.ensemble_primary_stem = ensemble_main_stem_pair[0]
        self.ensemble_secondary_stem = ensemble_main_stem_pair[2]
        self.is_normalization = root.is_normalization_var.get()
        self.is_wav_ensemble = root.is_wav_ensemble_var.get()
        self.wav_type_set = root.wav_type_set
        self.mp3_bit_set = root.mp3_bit_set_var.get()
        self.save_format = root.save_format_var.get()
        if not is_manual_ensemble:
            os.mkdir(self.ensemble_folder_name)

    def ensemble_outputs(self, audio_file_base, export_path, stem, is_4_stem=False, is_inst_mix=False):
        """Processes the given outputs and ensembles them with the chosen algorithm"""
        
        if is_4_stem:
            algorithm = root.ensemble_type_var.get()
            stem_tag = stem
        else:
            if is_inst_mix:
                algorithm = self.secondary_algorithm
                stem_tag = f"{self.ensemble_secondary_stem} {INST_STEM}"
            else:
                algorithm = self.primary_algorithm if stem == PRIMARY_STEM else self.secondary_algorithm
                stem_tag = self.ensemble_primary_stem if stem == PRIMARY_STEM else self.ensemble_secondary_stem

        stem_outputs = self.get_files_to_ensemble(folder=export_path, prefix=audio_file_base, suffix=f"_({stem_tag}).wav")
        audio_file_output = f"{self.is_testing_audio}{audio_file_base}{self.chosen_ensemble}_({stem_tag})"
        stem_save_path = os.path.join('{}'.format(self.main_export_path),'{}.wav'.format(audio_file_output))
        
        #print("get_files_to_ensemble: ", stem_outputs)
        
        if len(stem_outputs) > 1:
            spec_utils.ensemble_inputs(stem_outputs, algorithm, self.is_normalization, self.wav_type_set, stem_save_path, is_wave=self.is_wav_ensemble)
            save_format(stem_save_path, self.save_format, self.mp3_bit_set)
        
        if self.is_save_all_outputs_ensemble:
            for i in stem_outputs:
                save_format(i, self.save_format, self.mp3_bit_set)
        else:
            for i in stem_outputs:
                try:
                    os.remove(i)
                except Exception as e:
                    print(e)

    def ensemble_manual(self, audio_inputs, audio_file_base, is_bulk=False):
        """Processes the given outputs and ensembles them with the chosen algorithm"""
        
        is_mv_sep = True
        
        if is_bulk:
            number_list = list(set([os.path.basename(i).split("_")[0] for i in audio_inputs]))
            for n in number_list:
                current_list = [i for i in audio_inputs if os.path.basename(i).startswith(n)]
                audio_file_base = os.path.basename(current_list[0]).split('.wav')[0]
                stem_testing = "instrum" if "Instrumental" in audio_file_base else "vocals"
                if is_mv_sep:
                    audio_file_base = audio_file_base.split("_")
                    audio_file_base = f"{audio_file_base[1]}_{audio_file_base[2]}_{stem_testing}"
                self.ensemble_manual_process(current_list, audio_file_base, is_bulk)
        else:
            self.ensemble_manual_process(audio_inputs, audio_file_base, is_bulk)
            
    def ensemble_manual_process(self, audio_inputs, audio_file_base, is_bulk):
        
        algorithm = root.choose_algorithm_var.get()
        algorithm_text = "" if is_bulk else f"_({root.choose_algorithm_var.get()})"
        stem_save_path = os.path.join('{}'.format(self.main_export_path),'{}{}{}.wav'.format(self.is_testing_audio, audio_file_base, algorithm_text))
        spec_utils.ensemble_inputs(audio_inputs, algorithm, self.is_normalization, self.wav_type_set, stem_save_path, is_wave=self.is_wav_ensemble)
        save_format(stem_save_path, self.save_format, self.mp3_bit_set)

    def get_files_to_ensemble(self, folder="", prefix="", suffix=""):
        """Grab all the files to be ensembled"""
        
        return [os.path.join(folder, i) for i in os.listdir(folder) if i.startswith(prefix) and i.endswith(suffix)]

    def combine_audio(self, audio_inputs, audio_file_base):
        save_format_ = lambda save_path:save_format(save_path, root.save_format_var.get(), root.mp3_bit_set_var.get())
        spec_utils.combine_audio(audio_inputs, 
                                 os.path.join(self.main_export_path, f"{self.is_testing_audio}{audio_file_base}"), 
                                 self.wav_type_set,
                                 save_format=save_format_)

class AudioTools():
    def __init__(self, audio_tool):
        time_stamp = round(time.time())
        self.audio_tool = audio_tool
        self.main_export_path = Path(root.export_path_var.get())
        self.wav_type_set = root.wav_type_set
        self.is_normalization = root.is_normalization_var.get()
        self.is_testing_audio = f"{time_stamp}_" if root.is_testing_audio_var.get() else ''
        self.save_format = lambda save_path:save_format(save_path, root.save_format_var.get(), root.mp3_bit_set_var.get())
        self.align_window = TIME_WINDOW_MAPPER[root.time_window_var.get()]
        self.align_intro_val = INTRO_MAPPER[root.intro_analysis_var.get()]
        self.db_analysis_val = VOLUME_MAPPER[root.db_analysis_var.get()]
        self.is_save_align = root.is_save_align_var.get()#
        self.is_match_silence = root.is_match_silence_var.get()#
        self.is_spec_match = root.is_spec_match_var.get()
        
        self.phase_option = root.phase_option_var.get()#
        self.phase_shifts = PHASE_SHIFTS_OPT[root.phase_shifts_var.get()]
        
    def align_inputs(self, audio_inputs, audio_file_base, audio_file_2_base, command_Text, set_progress_bar):
        audio_file_base = f"{self.is_testing_audio}{audio_file_base}"
        audio_file_2_base = f"{self.is_testing_audio}{audio_file_2_base}"
        
        aligned_path = os.path.join('{}'.format(self.main_export_path),'{}_(Aligned).wav'.format(audio_file_2_base))
        inverted_path = os.path.join('{}'.format(self.main_export_path),'{}_(Inverted).wav'.format(audio_file_base))

        spec_utils.align_audio(audio_inputs[0], 
                               audio_inputs[1], 
                               aligned_path, 
                               inverted_path, 
                               self.wav_type_set, 
                               self.is_save_align, 
                               command_Text, 
                               self.save_format,
                               align_window=self.align_window,
                               align_intro_val=self.align_intro_val,
                               db_analysis=self.db_analysis_val,
                               set_progress_bar=set_progress_bar, 
                               phase_option=self.phase_option,
                               phase_shifts=self.phase_shifts,
                               is_match_silence=self.is_match_silence,
                               is_spec_match=self.is_spec_match)
        
    def match_inputs(self, audio_inputs, audio_file_base, command_Text):
        
        target = audio_inputs[0]
        reference = audio_inputs[1]
        
        command_Text(f"Processing... ")
        
        save_path = os.path.join('{}'.format(self.main_export_path),'{}_(Matched).wav'.format(f"{self.is_testing_audio}{audio_file_base}"))
        
        match.process(
            target=target,
            reference=reference,
            results=[match.save_audiofile(save_path, wav_set=self.wav_type_set),
            ],
        )
        
        self.save_format(save_path)
        
    def combine_audio(self, audio_inputs, audio_file_base):
        spec_utils.combine_audio(audio_inputs, 
                                 os.path.join(self.main_export_path, f"{self.is_testing_audio}{audio_file_base}"), 
                                 self.wav_type_set,
                                 save_format=self.save_format)
        
    def pitch_or_time_shift(self, audio_file, audio_file_base):
        is_time_correction = True
        rate = float(root.time_stretch_rate_var.get()) if self.audio_tool == TIME_STRETCH else float(root.pitch_rate_var.get())
        is_pitch = False if self.audio_tool == TIME_STRETCH else True
        if is_pitch:
            is_time_correction = True if root.is_time_correction_var.get() else False
        file_text = TIME_TEXT if self.audio_tool == TIME_STRETCH else PITCH_TEXT
        save_path = os.path.join(self.main_export_path, f"{self.is_testing_audio}{audio_file_base}{file_text}.wav")
        spec_utils.augment_audio(save_path, audio_file, rate, self.is_normalization, self.wav_type_set, self.save_format, is_pitch=is_pitch, is_time_correction=is_time_correction)

class MainWindow(TkinterDnD.Tk if is_dnd_compatible else tk.Tk):


    def assemble_model_data(self, model=None, arch_type=ENSEMBLE_MODE, is_dry_check=False, is_change_def=False, is_get_hash_dir_only=False):

        model_data: List[ModelData] = [ModelData(model, DEMUCS_ARCH_TYPE)]#

        return model_data
    
    
    def verify_audio(self, audio_file, is_process=True, sample_path=None):
        is_good = False
        error_data = ''
        
        if not type(audio_file) is tuple:
            audio_file = [audio_file]

        for i in audio_file:
            if os.path.isfile(i):
                try:
                    librosa.load(i, duration=3, mono=False, sr=44100) if not type(sample_path) is str else self.create_sample(i, sample_path)
                    is_good = True
                except Exception as e:
                    error_name = f'{type(e).__name__}'
                    traceback_text = ''.join(traceback.format_tb(e.__traceback__))
                    message = f'{error_name}: "{e}"\n{traceback_text}"'
                    if is_process:
                        audio_base_name = os.path.basename(i)
                        self.error_log_var.set(f'{ERROR_LOADING_FILE_TEXT[0]}:\n\n\"{audio_base_name}\"\n\n{ERROR_LOADING_FILE_TEXT[1]}:\n\n{message}')
                    else:
                        error_data = AUDIO_VERIFICATION_CHECK(i, message)

        if is_process:
            return is_good
        else:
            return is_good, error_data
      
    def create_sample(self, audio_file, sample_path=SAMPLE_CLIP_PATH):
        try:
            with audioread.audio_open(audio_file) as f:
                track_length = int(f.duration)
        except Exception as e:
            print('Audioread failed to get duration. Trying Librosa...')
            y, sr = librosa.load(audio_file, mono=False, sr=44100)
            track_length = int(librosa.get_duration(y=y, sr=sr))
        
        clip_duration = int(self.model_sample_mode_duration_var.get())
        
        if track_length >= clip_duration:
            offset_cut = track_length//3
            off_cut = offset_cut + track_length
            if not off_cut >= clip_duration:
                offset_cut = 0
            name_apped = f'{clip_duration}_second_'
        else:
            offset_cut, clip_duration = 0, track_length
            name_apped = ''

        sample = librosa.load(audio_file, offset=offset_cut, duration=clip_duration, mono=False, sr=44100)[0].T
        audio_sample = os.path.join(sample_path, f'{os.path.splitext(os.path.basename(audio_file))[0]}_{name_apped}sample.wav')
        sf.write(audio_sample, sample, 44100)
        
        return audio_sample


    #--Processing Methods-- 

    def process_check_wav_type(self):
        if self.wav_type_set_var.get() == '32-bit Float':
            self.wav_type_set = 'FLOAT'
        elif self.wav_type_set_var.get() == '64-bit Float':#
            self.wav_type_set = 'FLOAT' if not self.save_format_var.get() == WAV else 'DOUBLE'
        else:
            self.wav_type_set = self.wav_type_set_var.get()
            
    def process_preliminary_checks(self):
        """Verifies a valid model is chosen"""
        
        self.process_check_wav_type()
        
        if self.chosen_process_method_var.get() == ENSEMBLE_MODE:
            continue_process = lambda:False if len(self.ensemble_listbox_get_all_selected_models()) <= 1 else True
        if self.chosen_process_method_var.get() == VR_ARCH_PM:
            continue_process = lambda:False if self.vr_model_var.get() == CHOOSE_MODEL else True
        if self.chosen_process_method_var.get() == MDX_ARCH_TYPE:
            continue_process = lambda:False if self.mdx_net_model_var.get() == CHOOSE_MODEL else True
        if self.chosen_process_method_var.get() == DEMUCS_ARCH_TYPE:
            continue_process = lambda:False if self.demucs_model_var.get() == CHOOSE_MODEL else True

        return continue_process()

    def process_initialize(self):
        """Verifies the input/output directories are valid and prepares to thread the main process."""
        
        if not (
            self.chosen_process_method_var.get() == AUDIO_TOOLS 
            and self.chosen_audio_tool_var.get() in [ALIGN_INPUTS, MATCH_INPUTS] 
            and self.fileOneEntry_var.get() 
            and self.fileTwoEntry_var.get()
        ) and not (
            self.inputPaths and os.path.isfile(self.inputPaths[0])
        ):
            self.error_dialoge(INVALID_INPUT)
            return

            
        if not os.path.isdir(self.export_path_var.get()):
            self.error_dialoge(INVALID_EXPORT)
            return

        if not self.process_storage_check():
            return

        if self.chosen_process_method_var.get() != AUDIO_TOOLS:
            if not self.process_preliminary_checks():
                error_msg = INVALID_ENSEMBLE if self.chosen_process_method_var.get() == ENSEMBLE_MODE else INVALID_MODEL
                self.error_dialoge(error_msg)
                return
            target_function = self.process_start
        else:
            target_function = self.process_tool_start
        
        self.active_processing_thread = KThread(target=target_function)
        self.active_processing_thread.start()

    def process_tool_start(self):
        """Start the conversion for all the given mp3 and wav files"""

        def time_elapsed():
            return f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}'

        def get_audio_file_base(audio_file):
            if audio_tool.audio_tool == MANUAL_ENSEMBLE:
                return f'{os.path.splitext(os.path.basename(inputPaths[0]))[0]}'
            elif audio_tool.audio_tool in [ALIGN_INPUTS, MATCH_INPUTS]:
                return f'{os.path.splitext(os.path.basename(audio_file[0]))[0]}'
            else:
                return f'{os.path.splitext(os.path.basename(audio_file))[0]}'

        def handle_ensemble(inputPaths, audio_file_base):
            self.progress_bar_main_var.set(50)
            if self.choose_algorithm_var.get() == COMBINE_INPUTS:
                audio_tool.combine_audio(inputPaths, audio_file_base)
            else:
                audio_tool.ensemble_manual(inputPaths, audio_file_base)
            self.progress_bar_main_var.set(100)
            self.command_Text.write(DONE)

        def handle_alignment_match(audio_file, audio_file_base, command_Text, set_progress_bar):
            audio_file_2_base = f'{os.path.splitext(os.path.basename(audio_file[1]))[0]}'
            if audio_tool.audio_tool == MATCH_INPUTS:
                audio_tool.match_inputs(audio_file, audio_file_base, command_Text)
            else:
                command_Text(f"{PROCESS_STARTING_TEXT}\n")
                audio_tool.align_inputs(audio_file, audio_file_base, audio_file_2_base, command_Text, set_progress_bar)
            self.progress_bar_main_var.set(base * file_num)
            self.command_Text.write(f"{DONE}\n")

        def handle_pitch_time_shift(audio_file, audio_file_base):
            audio_tool.pitch_or_time_shift(audio_file, audio_file_base)
            self.progress_bar_main_var.set(base * file_num)
            self.command_Text.write(DONE)

        multiple_files = False
        stime = time.perf_counter()
        self.process_button_init()
        inputPaths = self.inputPaths
        is_verified_audio = True
        is_dual = False
        is_model_sample_mode = self.model_sample_mode_var.get()
        self.iteration = 0
        self.true_model_count = 1
        self.process_check_wav_type()
        process_complete_text = PROCESS_COMPLETE

        if self.chosen_audio_tool_var.get() in [ALIGN_INPUTS, MATCH_INPUTS]:
            if self.DualBatch_inputPaths:
                inputPaths = tuple(self.DualBatch_inputPaths)
            else:
                if not self.fileOneEntry_Full_var.get() or not self.fileTwoEntry_Full_var.get():
                    self.command_Text.write(NOT_ENOUGH_ERROR_TEXT)
                    self.process_end()
                    return
                else:
                    inputPaths = [(self.fileOneEntry_Full_var.get(), self.fileTwoEntry_Full_var.get())]

        try:
            total_files = len(inputPaths)
            if self.chosen_audio_tool_var.get() == TIME_STRETCH:
                audio_tool = AudioTools(TIME_STRETCH)
                self.progress_bar_main_var.set(2)
            elif self.chosen_audio_tool_var.get() == CHANGE_PITCH:
                audio_tool = AudioTools(CHANGE_PITCH)
                self.progress_bar_main_var.set(2)
            elif self.chosen_audio_tool_var.get() == MANUAL_ENSEMBLE:
                if self.chosen_audio_tool_var.get() == MANUAL_ENSEMBLE:
                    audio_tool = Ensembler(is_manual_ensemble=True)
                multiple_files = True
                if total_files <= 1:
                    self.command_Text.write(NOT_ENOUGH_ERROR_TEXT)
                    self.process_end()
                    return
            elif self.chosen_audio_tool_var.get() in [ALIGN_INPUTS, MATCH_INPUTS]:
                audio_tool = AudioTools(self.chosen_audio_tool_var.get())
                self.progress_bar_main_var.set(2)
                is_dual = True

            for file_num, audio_file in enumerate(inputPaths, start=1):
                self.iteration += 1
                base = (100 / total_files)
                audio_file_base = get_audio_file_base(audio_file)
                self.base_text = self.process_get_baseText(total_files=total_files, file_num=total_files if multiple_files else file_num, is_dual=is_dual)
                command_Text = lambda text: self.command_Text.write(self.base_text + text)

                set_progress_bar = lambda step, inference_iterations=0:self.process_update_progress(total_files=total_files, step=(step + (inference_iterations)))

                if not self.verify_audio(audio_file):
                    error_text_console = f'{self.base_text}"{os.path.basename(audio_file)}\" {MISSING_MESS_TEXT}\n'
                    if total_files >= 2:
                        self.command_Text.write(f'\n{error_text_console}')
                    is_verified_audio = False
                    continue

                audio_tool_action = audio_tool.audio_tool
                if audio_tool_action not in [MANUAL_ENSEMBLE, ALIGN_INPUTS, MATCH_INPUTS]:
                    audio_file = self.create_sample(audio_file) if is_model_sample_mode else audio_file
                    self.command_Text.write(f'{NEW_LINE if file_num != 1 else NO_LINE}{self.base_text}"{os.path.basename(audio_file)}\".{NEW_LINES}')
                elif audio_tool_action in [ALIGN_INPUTS, MATCH_INPUTS]:
                    text_write = ("File 1", "File 2") if audio_tool_action == ALIGN_INPUTS else ("Target", "Reference")
                    if audio_file[0] != audio_file[1]:
                        self.command_Text.write(f'{self.base_text}{text_write[0]}:  "{os.path.basename(audio_file[0])}"{NEW_LINE}')
                        self.command_Text.write(f'{self.base_text}{text_write[1]}:  "{os.path.basename(audio_file[1])}"{NEW_LINES}')
                    else:
                        self.command_Text.write(f'{self.base_text}{text_write[0]} & {text_write[1]} {SIMILAR_TEXT}{NEW_LINES}')
                        continue
                elif audio_tool_action == MANUAL_ENSEMBLE:
                    for n, i in enumerate(inputPaths):
                        self.command_Text.write(f'File {n+1} "{os.path.basename(i)}"{NEW_LINE}')
                    self.command_Text.write(NEW_LINE)
                    
                is_verified_audio = True

                if not audio_tool_action in [ALIGN_INPUTS, MATCH_INPUTS]:
                    command_Text(PROCESS_STARTING_TEXT)

                if audio_tool_action == MANUAL_ENSEMBLE:
                    handle_ensemble(inputPaths, audio_file_base)
                    break
                if audio_tool_action in [ALIGN_INPUTS, MATCH_INPUTS]:
                    process_complete_text = PROCESS_COMPLETE_2
                    handle_alignment_match(audio_file, audio_file_base, command_Text, set_progress_bar)
                if audio_tool_action in [TIME_STRETCH, CHANGE_PITCH]:
                    handle_pitch_time_shift(audio_file, audio_file_base)

            if total_files == 1 and not is_verified_audio:
                self.command_Text.write(f'{error_text_console}\n{PROCESS_FAILED}')
                self.command_Text.write(time_elapsed())
                playsound(FAIL_CHIME) if self.is_task_complete_var.get() else None
            else:
                self.command_Text.write('{}{}'.format(process_complete_text, time_elapsed()))
                playsound(COMPLETE_CHIME) if self.is_task_complete_var.get() else None

            self.process_end()

        except Exception as e:
            self.error_log_var.set(error_text(self.chosen_audio_tool_var.get(), e))
            self.command_Text.write(f'\n\n{PROCESS_FAILED}')
            self.command_Text.write(time_elapsed())
            playsound(FAIL_CHIME) if self.is_task_complete_var.get() else None
            self.process_end(error=e)

    def process_determine_secondary_model(self, process_method, main_model_primary_stem, is_primary_stem_only=False, is_secondary_stem_only=False):
        """Obtains the correct secondary model data for conversion."""
        
        secondary_model_scale = None
        secondary_model = tk.StringVar(value=NO_MODEL)
        
        if process_method == VR_ARCH_TYPE:
            secondary_model_vars = self.vr_secondary_model_vars
        if process_method == MDX_ARCH_TYPE:
            secondary_model_vars = self.mdx_secondary_model_vars
        if process_method == DEMUCS_ARCH_TYPE:
            secondary_model_vars = self.demucs_secondary_model_vars

        if main_model_primary_stem in [VOCAL_STEM, INST_STEM]:
            secondary_model = secondary_model_vars["voc_inst_secondary_model"]
            secondary_model_scale = secondary_model_vars["voc_inst_secondary_model_scale"].get()
        if main_model_primary_stem in [OTHER_STEM, NO_OTHER_STEM]:
            secondary_model = secondary_model_vars["other_secondary_model"]
            secondary_model_scale = secondary_model_vars["other_secondary_model_scale"].get()
        if main_model_primary_stem in [DRUM_STEM, NO_DRUM_STEM]:
            secondary_model = secondary_model_vars["drums_secondary_model"]
            secondary_model_scale = secondary_model_vars["drums_secondary_model_scale"].get()
        if main_model_primary_stem in [BASS_STEM, NO_BASS_STEM]:
            secondary_model = secondary_model_vars["bass_secondary_model"]
            secondary_model_scale = secondary_model_vars["bass_secondary_model_scale"].get()

        if secondary_model_scale:
           secondary_model_scale = float(secondary_model_scale)

        if not secondary_model.get() == NO_MODEL:
            secondary_model = ModelData(secondary_model.get(), 
                                        is_secondary_model=True, 
                                        primary_model_primary_stem=main_model_primary_stem, 
                                        is_primary_model_primary_stem_only=is_primary_stem_only, 
                                        is_primary_model_secondary_stem_only=is_secondary_stem_only)
            if not secondary_model.model_status:
                secondary_model = None
        else:
            secondary_model = None
            
        return secondary_model, secondary_model_scale
        
    def process_determine_demucs_pre_proc_model(self, primary_stem=None):
        """Obtains the correct pre-process secondary model data for conversion."""
        
        # Check if a pre-process model is set and it's not the 'NO_MODEL' value
        if self.demucs_pre_proc_model_var.get() != NO_MODEL and self.is_demucs_pre_proc_model_activate_var.get():
            pre_proc_model = ModelData(self.demucs_pre_proc_model_var.get(), 
                                        primary_model_primary_stem=primary_stem, 
                                        is_pre_proc_model=True)
            
            # Return the model if it's valid
            if pre_proc_model.model_status:
                return pre_proc_model
                
        return None

    def process_determine_vocal_split_model(self):
        """Obtains the correct vocal splitter secondary model data for conversion."""
        
        # Check if a vocal splitter model is set and if it's not the 'NO_MODEL' value
        if self.set_vocal_splitter_var.get() != NO_MODEL and self.is_set_vocal_splitter_var.get():
            vocal_splitter_model = ModelData(self.set_vocal_splitter_var.get(), is_vocal_split_model=True)
            
            # Return the model if it's valid
            if vocal_splitter_model.model_status:
                return vocal_splitter_model
                
        return None

    def check_only_selection_stem(self, checktype):
        
        chosen_method = self.chosen_process_method_var.get()
        is_demucs = chosen_method == DEMUCS_ARCH_TYPE#

        stem_primary_label = self.is_primary_stem_only_Demucs_Text_var.get() if is_demucs else self.is_primary_stem_only_Text_var.get()
        stem_primary_bool = self.is_primary_stem_only_Demucs_var.get() if is_demucs else self.is_primary_stem_only_var.get()
        stem_secondary_label = self.is_secondary_stem_only_Demucs_Text_var.get() if is_demucs else self.is_secondary_stem_only_Text_var.get()
        stem_secondary_bool = self.is_secondary_stem_only_Demucs_var.get() if is_demucs else self.is_secondary_stem_only_var.get()

        if checktype == VOCAL_STEM_ONLY:
            return not (
                (not VOCAL_STEM_ONLY == stem_primary_label and stem_primary_bool) or 
                (not VOCAL_STEM_ONLY in stem_secondary_label and stem_secondary_bool)
            )
        elif checktype == INST_STEM_ONLY:
            return (
                (INST_STEM_ONLY == stem_primary_label and stem_primary_bool and self.is_save_inst_set_vocal_splitter_var.get() and self.set_vocal_splitter_var.get() != NO_MODEL) or 
                (INST_STEM_ONLY == stem_secondary_label and stem_secondary_bool and self.is_save_inst_set_vocal_splitter_var.get() and self.set_vocal_splitter_var.get() != NO_MODEL)
            )
        elif checktype == IS_SAVE_VOC_ONLY:
            return (
                (VOCAL_STEM_ONLY == stem_primary_label and stem_primary_bool) or 
                (VOCAL_STEM_ONLY == stem_secondary_label and stem_secondary_bool)
            )
        elif checktype == IS_SAVE_INST_ONLY:
            return (
                (INST_STEM_ONLY == stem_primary_label and stem_primary_bool) or 
                (INST_STEM_ONLY == stem_secondary_label and stem_secondary_bool)
            )

    def determine_voc_split(self, models):
        is_vocal_active = self.check_only_selection_stem(VOCAL_STEM_ONLY) or self.check_only_selection_stem(INST_STEM_ONLY)

        if self.set_vocal_splitter_var.get() != NO_MODEL and self.is_set_vocal_splitter_var.get() and is_vocal_active:
            model_stems_list = self.model_list(VOCAL_STEM, INST_STEM, is_dry_check=True, is_check_vocal_split=True)
            if any(model.model_basename in model_stems_list for model in models):
                return 1
        
        return 0
        
    def process_start(self, inputPath):
        """Start the conversion for all the given mp3 and wav files"""
        
        # try:
           
        model = self.assemble_model_data("demucs_model_var", DEMUCS_ARCH_TYPE)

        
        true_model_4_stem_count = sum(m.demucs_4_stem_added_count if m.process_method == DEMUCS_ARCH_TYPE else 0 for m in model)
        true_model_pre_proc_model_count = sum(2 if m.pre_proc_model_activated else 0 for m in model)
        self.true_model_count = sum(2 if m.is_secondary_model_activated else 1 for m in model) + true_model_4_stem_count + true_model_pre_proc_model_count + self.determine_voc_split(model)

        #print("self.true_model_count", self.true_model_count)

        audio_file = inputPath

        # if self.verify_audio(audio_file):
        #     audio_file = self.create_sample(audio_file) if is_model_sample_mode else audio_file


        for current_model_num, current_model in enumerate(model, start=1):
            self.iteration += 1

            if is_ensemble:
                self.command_Text.write(f'Ensemble Mode - {current_model.model_basename} - Model {current_model_num}/{len(model)}{NEW_LINES}')

            model_name_text = f'({current_model.model_basename})' if not is_ensemble else ''
            self.command_Text.write(base_text + f'{LOADING_MODEL_TEXT} {model_name_text}...')

            set_progress_bar = lambda step, inference_iterations=0:self.process_update_progress(total_files=inputPath_total_len, step=(step + (inference_iterations)))
            write_to_console = lambda progress_text, base_text=base_text:self.command_Text.write(base_text + progress_text)

            audio_file_base = f"{file_num}_{os.path.splitext(os.path.basename(audio_file))[0]}"
            audio_file_base = audio_file_base if not self.is_testing_audio_var.get() or is_ensemble else f"{round(time.time())}_{audio_file_base}"
            audio_file_base = audio_file_base if not is_ensemble else f"{audio_file_base}_{current_model.model_basename}"
            if not is_ensemble:
                audio_file_base = audio_file_base if not self.is_add_model_name_var.get() else f"{audio_file_base}_{current_model.model_basename}"

            if self.is_create_model_folder_var.get() and not is_ensemble:
                export_path = os.path.join(Path(self.export_path_var.get()), current_model.model_basename, os.path.splitext(os.path.basename(audio_file))[0])
                if not os.path.isdir(export_path):os.makedirs(export_path) 

            process_data = {
                            'model_data': current_model, 
                            'export_path': export_path,
                            'audio_file_base': audio_file_base,
                            'audio_file': audio_file,
                            'set_progress_bar': set_progress_bar,
                            'write_to_console': write_to_console,
                            'process_iteration': self.process_iteration,
                            'cached_source_callback': self.cached_source_callback,
                            'cached_model_source_holder': self.cached_model_source_holder,
                            'list_all_models': self.all_models,
                            'is_ensemble_master': False,
                            'is_4_stem_ensemble': True if self.ensemble_main_stem_var.get() in [FOUR_STEM_ENSEMBLE, MULTI_STEM_ENSEMBLE] and is_ensemble else False}
            
            
            seperator = SeperateDemucs(current_model, process_data)
                
            seperator.seperate()
            
            
    
            
        clear_gpu_cache()




def extract_stems(audio_file_base, export_path):
    
    filenames = [file for file in os.listdir(export_path) if file.startswith(audio_file_base)]

    pattern = r'\(([^()]+)\)(?=[^()]*\.wav)'
    stem_list = []

    for filename in filenames:
        match = re.search(pattern, filename)
        if match:
            stem_list.append(match.group(1))
            
    counter = Counter(stem_list)
    filtered_lst = [item for item in stem_list if counter[item] > 1]

    return list(set(filtered_lst))


