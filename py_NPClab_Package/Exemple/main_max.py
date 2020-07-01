# -*-coding:utf-8 -*-
import numpy as np
import pandas as pd
from py_NPClab_Package.utilitaire_load.basic_load import LabviewFilesReward, LabviewFilesTrajectory, LoadData
from py_NPClab_Package.utilitaire_traitement.TrajectoryTraitement import BasicTraitmentTrajectory
from py_NPClab_Package.traitement_labview.Labview_traitment import AnalyseFromLabview


#------------------------------------- parti import data labview ---------------------------------------------------

dir_txt_traj = r'Y:\Analyse_maxime\cplx10\fichier_traj\*.txt'

# chargement de la trajectoire contient "x, y, point"
trajectoire = LoadData.init_data(LabviewFilesTrajectory, dir_txt_traj)
# chargement des temps en ms "temps" contenu dans le fichier rewards
reward = LoadData.init_data(LabviewFilesReward, dir_txt_traj)

# traitement
data_AFL = AnalyseFromLabview()
# création d'un dataframe contenant "trajectoire_x, trajectoire_y, reward, rewards, omissions"
data_tracking_AFL = data_AFL.load_data_from_labview(trajectoire, reward)

# ------------------------------------- parti traitement de la trajectoire labview -----------------
traitment_AFL = BasicTraitmentTrajectory()
data_traiter_AFL = traitment_AFL.correction(data_AFL, data_AFL.format_correction)

# ------------------------------------- parti preparation des données pour le plot ---------
from py_NPClab_Package.utilitaire_traitement.PreFormatData import PreFormatData

data_formater = PreFormatData()

omission_time = pd.Series(data_AFL.omission, name='omission')

# ---------------------------------------------
data_formater.make_event_around(data=traitment_AFL.norme_vecteur, reward=omission_time)

# ------------------------------------- parti plot ----------------------------------------
# from py_NPClab_Package.utilitaire_plot.TrajectoireBasicPlot import SpecifiquePlot
# plotcomportement = SpecifiquePlot()
# plotcomportement.plot_event_around(['trajectoire'], data_formater.around_event)
# plotcomportement.plot_norme_vecteur(traitment_AFL.norme_vecteur, data_AFL)
# plotcomportement.plot_traj(data_tracking_AFL, data_AFL.couple_de_points[0],'AFL')

# -------------------------------------------- partie event ---------------------------------------------------
from py_NPClab_Package.traitement_event.EventTraitement import EventRewardNeuralynx
from py_NPClab_Package.utilitaire_load.basic_load import EventFilesSerialiser, LoadData

dir_save: str = r'Y:\Analyse_maxime\cplx10\save'
dir_data: str = r'Y:\Analyse_maxime\cplx10'
dir_profile_pattern: str = r'Y:\Analyse_maxime\profile_pattern'


name = 'all_event'
all_event = LoadData.init_data(EventFilesSerialiser, dir_save, name)

reward = EventRewardNeuralynx()

profile_pattern_synchro = reward.get_profile(dir_pattern=dir_profile_pattern, name_pattern='synchro_video_classic')
profile_pattern_stim = reward.get_profile(dir_pattern=dir_profile_pattern, name_pattern='stimulation_classic')



start_stim, start_stim_index = reward.set_reward(reward_time=all_event.data['num segment : 0']['time'],
                                                 reward_index=all_event.data['num segment : 0']['index'],
                                                 profile_pattern_synchro=profile_pattern_synchro,
                                                 profile_pattern_stim=profile_pattern_stim)
start_stim_index = start_stim_index.astype(dtype=int)


# -------------------------------------------- partie reload specifique neurone ---------------------------------
from py_NPClab_Package.utilitaire_load.basic_load import NeuroneFilesSerialiser

dir_save = r'Y:\Analyse_maxime\cplx10\save'
name = 'segment0_neurone0'
neurone = LoadData.init_data(NeuroneFilesSerialiser, dir_save, name)


# ------------------------------------- parti reload specification segment ----------------------------------
from py_NPClab_Package.utilitaire_load.basic_load import SegmentBTFilesSerialiser

dir_save = r'Y:\Analyse_maxime\cplx10\save'
name = 'segment_infos'
num_segment = 0
base_time_segment = LoadData.init_data(SegmentBTFilesSerialiser, dir_save, name, num_segment)

# -------------------------------------- parti recallage omission sur temps spike --------------------------
from py_NPClab_Package.traitement_event.EventTraitement import EventFromOther

omission_recallage = EventFromOther()
omission_recallage.start(np.array(omission_time), base_time_segment.data, reward.reward_time_ref_rmz)
omission_spike_time = pd.Series(base_time_segment.data[omission_recallage.event_index_in_raw])

# ----------------------------- nettoyage du neurone autour de la stim -------------------------------------
# dir_data = r'Y:\Analyse_maxime\cplx10'
# new_spike = CleaningSpikeTime(dir_data=dir_data, name_neurone='segment0_neurone0',
#                               start_stim=start_stim, time_ref_synchro=reward.reward_time_ref_rmz)
# neurone_spike_bool = new_spike.load_neurone()

# new_omission = CleaningSpikeTime(dir_data=dir_data, name_neurone='segment1_neurone0',
#                               start_stim=omission_spike_time, time_ref_synchro=reward.reward_time_ref_rmz)
# neurone_spike_omission_bool = new_omission.load_other()

# ------------------------------------- parti preparation des données pour le plot avec spike---------
from py_NPClab_Package.utilitaire_plot.BasicPlotSpike import GenericPlotV2
from py_NPClab_Package.utilitaire_load.basic_load import ImportNeuralynx

raw_brute = LoadData.init_data(ImportNeuralynx, dir_data, 'csc')

plot = GenericPlotV2()


reward_spike_time = pd.Series(base_time_segment.data[start_stim_index])
reward_plot_data = plot.plot_raster_event_spike(spike_time=np.array(neurone.data['time']),
                                                time_event=reward_spike_time,
                                                name='reward')
plot.plot_kernel_density(reward_plot_data, 'reward')
# -------------------------------------------- partie plot spike ----------------------------------------------
from py_NPClab_Package.utilitaire_load.basic_load import NeuroneFilesSerialiser

# dir_save = r'Y:\python\import_neuralynxv2\data\cplx07 + bsl\save'
# name = 'segment1_neurone0'
# neurone0 = LoadData.init_data(NeuroneFilesSerialiser, dir_save, name)

# plot.plot_frequence_glissante(neurones=[np.array(neurone_spike_bool['time'].dropna())],
#                               name_neurone=['neurone'],
#                               taille_fenetre=15, pas_de_gliss=5, name='neuron')
# plot.plot_burst_glissant(neurones_times=[neurone_spike_bool['time'].dropna()], neurones_isi=[neurone_spike_bool['isi'].dropna()],
#                               name_neurone=['neurone'],
#                               taille_fenetre=15, pas_de_gliss=5, name='neuron')
#
# plot.plotcorrelogram(neurones=[np.array(neurone_spike_bool['time'].dropna())], lag_max=0.5, lenght_of_bin=0.001, name='cross')

intervalle = [1 * 32000, 18 * 32000]  # intervalle par seconde

# plot.plot_neurone_in_raw(intervalle=intervalle_omission, neurones_index_val=[neurone_spike_omission_bool['time_index_in raw'].dropna().astype(dtype=int)], raw_signal=raw_brute.signal_segments['num segment : 1'],
#                          event=all_event, option_other_trigger=omission_recallage.event_index_in_raw,
#                          name='raw plot', option_csc=['CSC2','CSC6','CSC7','CSC10'])

plot.plot_neurone_in_raw(intervalle=intervalle, neurones_index_val=[neurone_spike_bool['time_index_in raw'].dropna().astype(dtype=int)],
                         raw_signal=raw_brute.signal_segments['num segment : 0'],
                         event=all_event['num segment : 0'], name='raw plot', option_csc=['CSC2', 'CSC6', 'CSC7', 'CSC8'])

plot.plot_waveform_neurone(intervalle=intervalle, neurones_index_val=[neurone_spike_bool['time_index_in raw'].dropna().astype(dtype=int)],
                           raw_signal=raw_brute.signal_segments['num segment : 1'],
                         event=all_event,
                         name='raw plot', option_csc=['CSC2','CSC6', 'CSC7', 'CSC8'])