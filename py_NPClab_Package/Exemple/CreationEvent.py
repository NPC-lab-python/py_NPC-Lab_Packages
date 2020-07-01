from py_NPClab_Package.utilitaire_load.basic_load import ImportNeuralynx, LoadData
from py_NPClab_Package.traitement_event.EventTraitement import EventFileNeuralynx

class EventInit(object):
    def __init__(self, dir_data: str):
        self.dir_data = dir_data

    def set_event(self):
        event_brute = LoadData.init_data(ImportNeuralynx, self.dir_data, 'event')
        Event = EventFileNeuralynx(dir_data=self.dir_data, event=event_brute)
        info_event, all_event = Event.set_event()

if __name__ == '__main__':
    dir_data: str = r'Y:\Analyse_maxime\cplx10'
    event = EventInit(dir_data=dir_data)
    event.set_event()