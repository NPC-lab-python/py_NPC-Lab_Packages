from py_NPClab_Package.utilitaire_load.basic_load import ImportNeuralynx, LoadData
from py_NPClab_Package.traitement_event.EventTraitement import EventFileNeuralynx, EventRewardNeuralynx


dir_data: str = r'Y:\Analyse_maxime\cplx10'
dir_profile_pattern: str = r'Y:\Analyse_maxime\profile_pattern'

event_brute = LoadData.init_data(ImportNeuralynx, dir_data, 'event')

Event = EventFileNeuralynx(dir_data=dir_data, event=event_brute)
info_event, all_event = Event.set_event()