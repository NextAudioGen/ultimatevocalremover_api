import platform

#Platform Details
OPERATING_SYSTEM = platform.system()
SYSTEM_ARCH = platform.platform()
SYSTEM_PROC = platform.processor()
ARM = 'arm'

# Network Constants
N_BINS = 'n_bins'


ALL_STEMS = 'All Stems'
VOCAL_STEM = 'Vocals'
INST_STEM = 'Instrumental'
OTHER_STEM = 'Other'
BASS_STEM = 'Bass'
DRUM_STEM = 'Drums'
GUITAR_STEM = 'Guitar'
PIANO_STEM = 'Piano'
SYNTH_STEM = 'Synthesizer'
STRINGS_STEM = 'Strings'
WOODWINDS_STEM = 'Woodwinds'
BRASS_STEM = 'Brass'
WIND_INST_STEM = 'Wind Inst'
NO_OTHER_STEM = 'No Other'
NO_BASS_STEM = 'No Bass'
NO_DRUM_STEM = 'No Drums'
NO_GUITAR_STEM = 'No Guitar'
NO_PIANO_STEM = 'No Piano'
NO_SYNTH_STEM = 'No Synthesizer'
NO_STRINGS_STEM = 'No Strings'
NO_WOODWINDS_STEM = 'No Woodwinds'
NO_WIND_INST_STEM = 'No Wind Inst'
NO_BRASS_STEM = 'No Brass'
PRIMARY_STEM = 'Primary Stem'
SECONDARY_STEM = 'Secondary Stem'


NO_STEM = "No "

NON_ACCOM_STEMS = (
            VOCAL_STEM,
            OTHER_STEM,
            BASS_STEM,
            DRUM_STEM,
            GUITAR_STEM,
            PIANO_STEM,
            SYNTH_STEM,
            STRINGS_STEM,
            WOODWINDS_STEM,
            BRASS_STEM,
            WIND_INST_STEM)