from py_NPClab_Package.utilitaire_load.basic_load import ImportNeuralynx, LoadData
from py_NPClab_Package.traitement_event.EventTraitement import EventFileNeuralynx, EventRewardNeuralynx


dir_data: str = r'Y:\Analyse_maxime\cplx10'
dir_profile_pattern: str = r'Y:\Analyse_maxime'

event_brute = LoadData.init_data(ImportNeuralynx, dir_data, 'event')

Event = EventFileNeuralynx(dir_data=dir_data, event=event_brute)
info_event, all_event = Event.set_event()

reward = EventRewardNeuralynx()

#Si le dossier "profile_pattern" n'existe pas , il sera créé.

# reward.set_profile(dir_profile_pattern=dir_data, time_down=0.25, time_up=0.25,
#                          pattern='CA', name_profile='synchro_video_spe')

reward.set_profile(time_down=0.0475, time_up=0.0025,
                        pattern='CABABABABA', name_profile='synchro_video_classic', other_path=dir_profile_pattern)

reward.set_profile(time_down=0.005, time_up=0.005,
                       pattern='AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA',
                   name_profile='stimulation_classic', other_path=dir_profile_pattern)

# reward.set_profile(dir_profile_pattern=dir_data, time_down=0.9, time_up=0.1,
#                      pattern='ABABABABABABABABABA', name='protocole_opto_1')

print('t')