# -*-coding:utf-8 -*-
import numpy as np
import pandas as pd
from NPClab_Package.utilitaire_load.basic_load import LabviewFilesReward, LabviewFilesTrajectory, LoadData
from NPClab_Package.utilitaire_traitement.TrajectoryTraitement import BasicTraitmentTrajectory
from NPClab_Package.traitement_labview.Labview_traitment import AnalyseFromLabview
dir_data: str = r'Y:\python\import_neuralynxv2\data\Guismo 23 equdif 9_probleme'
dir_save = r'Y:\python\import_neuralynxv2\data\Guismo 23 equdif 9_probleme\save'


#------------------------------------- parti import data labview ---------------------------------------------------

dir_txt_traj = r'Y:\python\import_neuralynxv2\data\Guismo 23 equdif 9_probleme\fichier_traj\*.txt'

# chargement de la trajectoire contient "x, y, point"
trajectoire = LoadData.init_data(LabviewFilesTrajectory, dir_txt_traj)
# chargement des temps en ms "temps" contenu dans le fichier rewards
reward_time_from_txt = LoadData.init_data(LabviewFilesReward, dir_txt_traj)

# traitement
data_AFL = AnalyseFromLabview()
# création d'un dataframe contenant "trajectoire_x, trajectoire_y, reward_time_from_txt, rewards, omissions"
data_tracking_AFL = data_AFL.load_data_from_labview(trajectoire, reward_time_from_txt)

# ------------------------------------- parti traitement de la trajectoire labview -----------------
traitment_AFL = BasicTraitmentTrajectory()
# data_AFL_brute = traitment_AFL.data_brute(data_AFL, data_AFL.format_correction)
data_traiter_AFL = traitment_AFL.correction(data_AFL, data_AFL.format_correction)

# ------------------------------------- parti preparation des données pour le plot ---------
from NPClab_Package.utilitaire_traitement.PreFormatData import PreFormatData

data_formater = PreFormatData()

omission_time = pd.Series(data_AFL.omission, name='omission')
# ---------------------------------------------
data_formater.make_event_around(data=traitment_AFL.norme_vecteur, reward=omission_time)



# -------------------------------------------- partie event ---------------------------------------------------
from NPClab_Package.utilitaire_load.basic_load import ImportNeuralynx, LoadData
from NPClab_Package.traitement_event.EventTraitement import EventFileNeuralynx, EventRewardNeuralynx


# dir_data: str = r'Y:\python\import_neuralynxv2\data\Session pour Steve'

event_brute = LoadData.init_data(ImportNeuralynx, dir_data, 'event')

Event = EventFileNeuralynx(dir_data=dir_data, event=event_brute)
info_event, all_event = Event.set_event()
dir_pattern = r'Y:\python\import_neuralynxv2\data\Guismo 23 equdif 9_probleme\profile_pattern'

reward_time_detect = EventRewardNeuralynx()

# -------------------------------------- création des profiles de pattern -------------------------------------

# dir_pattern = r'Y:\python\import_neuralynxv2\data\Guismo 29 equdif 9_ref_all_ok\profile_pattern'
#Si le dossier "profile_pattern" n'existe pas , il sera créé.
reward_time_detect.set_profile(dir_profile_pattern=dir_data, time_down=0.25, time_up=0.25,
                         pattern='CA', name_profile='synchro_video_spe')
# reward_time_detect.set_profile(dir_profile_pattern=dir_data, time_down=0.0475, time_up=0.0025,
#                        pattern='CABABABABA', name_profile='synchro_video_classic')
reward_time_detect.set_profile(dir_profile_pattern=dir_data, time_down=0.005, time_up=0.005,
                       pattern='AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA', name_profile='stimulation_classic')

profile_synchro = reward_time_detect.get_profile(dir_pattern=dir_pattern, name_pattern='synchro_video_spe')
profile_stim = reward_time_detect.get_profile(dir_pattern=dir_pattern, name_pattern='stimulation_classic')

# ------------------------------------ event
start_stim, start_stim_index = reward_time_detect.set_reward(reward_time_from_txt=reward_time_from_txt.data,
    reward_time=all_event['num segment : 0']['time'],
    reward_index=all_event['num segment : 0']['index'], profile_pattern_stim=profile_stim, profile_pattern_synchro=profile_synchro)
start_stim_index = start_stim_index.astype(dtype=int)

# ------------------------------------ parti création des informations experiences (segment) -----------------------------------------
from NPClab_Package.traitement_spike.NeuroneTraitment import Segment

segment_infos = Segment(dir_data=dir_data)
segment_data = segment_infos.set_segment(event_brute=event_brute.Event, event_final=all_event)

del segment_infos, segment_data

# ------------------------------------- parti reload specification segment ----------------------------------
from NPClab_Package.utilitaire_load.basic_load import SegmentFilesSerialiser

# dir_save = r'Y:\python\import_neuralynxv2\data\Session pour Steve\save'
name = 'segment_infos'
num_segment = 0
base_time_segment = LoadData.init_data(SegmentFilesSerialiser, dir_save, name, num_segment)

# -------------------------------------- parti recallage omission sur temps rawsignal "csc" --------------------------
from NPClab_Package.traitement_event.EventTraitement import EventFromOther

omission_recallage = EventFromOther()
omission_recallage.start(np.array(omission_time), base_time_segment.data, reward_time_detect.reward_time_ref_rmz)
omission_spike_time = pd.Series(base_time_segment.data[omission_recallage.event_index_in_raw])


# ------------------------------------- parti plot ----------------------------------------
from NPClab_Package.utilitaire_plot.TrajectoireBasicPlot import SpecifiquePlot
plotcomportement = SpecifiquePlot()
plotcomportement.plot_event_around(['trajectoire'], data_formater.around_event)
plotcomportement.plot_norme_vecteur(traitment_AFL.norme_vecteur, data_AFL)
plotcomportement.plot_traj(data_tracking_AFL, data_AFL.couple_de_points[0], "AFL")
plotcomportement.plot_norme_vecteur(traitment_AFL.acceleration_smooth, data_AFL)

# ------------------------ save divers data in txt --------------------------
from NPClab_Package.utlilitaire_saving.Saving_traitment import SavingMethodes

# save = SavingMethodes()
# save.save_data_text(data=np.array(omission_time)*1000,
#                     name='omission_time',
#                     path=r'Y:\python\import_neuralynxv2\data\Guismo 29 equdif 9_ref_all_ok')

# save.save_data_text(data=np.array([reward_time_detect.reward_time_ref_rmz]),
#                     name='lag_time',
#                     path=r'Y:\python\import_neuralynxv2\data\Guismo 29 equdif 9_ref_all_ok')

# save.save_data_text(data=np.array(data_AFL.reward),
#                     name='reward_time_from_trajectoire',
#                     path=r'Y:\python\import_neuralynxv2\data\Guismo 29 equdif 9_ref_all_ok')

# save.save_data_text(data=np.array(data_AFL.reward_from_txt),
#                     name='reward_time_from_txt_around',
#                     path=r'Y:\python\import_neuralynxv2\data\Guismo 29 equdif 9_ref_all_ok')

# save.save_data_text(data=np.array(reward_time_detect.start_stim),
#                     name='reward_time_in_neuralynx',
#                     path=r'Y:\python\import_neuralynxv2\data\Guismo 29 equdif 9_ref_all_ok')

# save.save_data_text(data=np.array(omission_spike_time),
#                     name='omission_time_in_neuralynx',
#                     path=r'Y:\python\import_neuralynxv2\data\Guismo 29 equdif 9_ref_all_ok')

# --------------------------
start_stim-reward_time_detect.reward_time_ref_rmz
