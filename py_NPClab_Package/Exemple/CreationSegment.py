
from py_NPClab_Package.utilitaire_load.basic_load import ImportNeuralynx, LoadData
from py_NPClab_Package.utilitaire_load.basic_load import EventFilesSerialiser
from py_NPClab_Package.traitement_spike.NeuroneTraitment import Segment

class SegmentInit(object):
    def __init__(self, dir_save: str, dir_data: str):
        self.dir_save = dir_save
        self.dir_data = dir_data

    def set_segment(self):
        event_brute = LoadData.init_data(ImportNeuralynx, self.dir_data, 'event')
        name = 'all_event'
        all_event = LoadData.init_data(EventFilesSerialiser, self.dir_save, name)
        segment_infos = Segment(dir_data=self.dir_data)
        segment_data = segment_infos.set_segment(event_brute=event_brute.Event, event_final=all_event.data)

if __name__ == '__main__':
    dir_save: str = r'Y:\Analyse_maxime\cplx10\save'
    dir_data: str = r'Y:\Analyse_maxime\cplx10'
    segment = SegmentInit(dir_data=dir_data, dir_save=dir_save)
    segment.set_segment()