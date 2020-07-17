from py_NPClab_Package.utilitaire_load.basic_load import LoadData, NeuralynxFilesSpike, SegmentFilesSerialiser, EventFilesSerialiser
from py_NPClab_Package.traitement_spike.NeuroneTraitment import Spike, PreFormatSpike
import sys
from typing import List, Dict


class NeuroneInit(object):

    def __init__(self, dir_save: str, dir_data: str, dir_spikefile: str):
        self.dir_save = dir_save
        self.dir_data = dir_data
        self.dir_spikefile = dir_spikefile

    def set_neurone(self, neurones: List[str]):
        """
        Cette méthode permet la création du fichier intermédiaire du neurone dans le segment voulu.

        """
        name_event = 'all_event'
        all_event = LoadData.init_data(EventFilesSerialiser, self.dir_save, name_event)

        name_seg = 'segment_infos'
        segment_data = LoadData.init_data(SegmentFilesSerialiser, self.dir_save, name_seg)

        spikefiles = LoadData.init_data(NeuralynxFilesSpike, self.dir_spikefile)

        try:
            assert len(spikefiles.data) == len(neurones)
        except AssertionError:
            print(f' Il y a un problème entre le nombre de neurone entrés et existants')
            sys.exit()
        """
        Pour pouvoir traiter les neurones,
            il faut :
            - spikefiles
            - all_event
            _ segment_data
            - nom du neurone pré-traité présent dans le dossier "save" à traiter, ex : "neurone0_brute"
            - le num_segment correspondant à l'experimentation
        """

        preparation_spike = PreFormatSpike(dir_data=self.dir_data, spike_files=spikefiles,
                                           segment_infos=segment_data.data,
                                           all_event=all_event.data)
        preparation_spike.set_spike()

        """
        
        """
        for i in neurones:
            for idx in list(segment_data.data.keys()):
                spike = Spike(dir_data=self.dir_data, name_neurone=i, segment_infos=segment_data.data, num_segment=idx[-1])
                spike.set_neurone()
                del spike
        del segment_data, preparation_spike


if __name__ == '__main__':

    dir_save: str = r'Y:\Analyse_maxime\cplx10\save'
    dir_data: str = r'Y:\Analyse_maxime\cplx10'
    dir_spikefile: str = r'Y:\Analyse_maxime\cplx10\clustering\*.txt'

    neurones = NeuroneInit(dir_data=dir_data, dir_save=dir_save, dir_spikefile=dir_spikefile)
    neurones.set_neurone(neurones=['neurone0_brute', 'neurone1_brute'], num_segment=0)