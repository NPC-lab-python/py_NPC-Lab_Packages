from py_NPClab_Package.utilitaire_load.basic_load import ImportNeuralynx, LoadData

# ------------------------------------- parti reload specification segment ----------------------------------
from py_NPClab_Package.utilitaire_load.basic_load import SegmentFilesSerialiser, EventFilesSerialiser

dir_save = r'Y:\Analyse_maxime\cplx10\save'
name = 'segment_infos'
segment_data = LoadData.init_data(SegmentFilesSerialiser, dir_save, name)

name = 'all_event'
all_event = LoadData.init_data(EventFilesSerialiser, dir_save, name)

from py_NPClab_Package.traitement_spike.NeuroneTraitment import Segment, Spike, PreFormatSpike, CleaningSpikeTime
from py_NPClab_Package.utilitaire_load.basic_load import NeuralynxFilesSpike

dir_spikefile: str = r'Y:\Analyse_maxime\cplx10\clustering\*.txt'
dir_data: str = r'Y:\Analyse_maxime\cplx10'


spikefiles = LoadData.init_data(NeuralynxFilesSpike, dir_spikefile)

"""
Pour pouvoir traiter les neurones,
    il faut :
    - spikefiles
    - all_event
    _ segment_data
    - nom du neurone pré-traité présent dans le dossier "save" à traiter, ex : "neurone0_brute"
    - le num_segment correspondant à l'experimentation
"""

e = PreFormatSpike(dir_data=dir_data, spike_files=spikefiles, segment_infos=segment_data.data, all_event=all_event)
e.set_spike()

spike = Spike(dir_data=dir_data, name_neurone='neurone0_brute', segment_infos=segment_data.data, num_segment=0)
spike.set_neurone()
spike = Spike(dir_data=dir_data, name_neurone='neurone1_brute', segment_infos=segment_data.data, num_segment=0)
spike.set_neurone()
# spike = Spike(dir_data=dir_data, name_neurone='neurone1_brute', segment_infos=segment_data, num_segment=0)
# spike.set_neurone()
# spike = Spike(dir_data=dir_data, name_neurone='neurone1_brute', segment_infos=segment_data, num_segment=1)
# spike.set_neurone()

del spike, segment_data, e
