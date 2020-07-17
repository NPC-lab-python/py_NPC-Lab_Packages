# -*-coding:utf-8 -*-
import numpy as np
import pandas as pd
from py_NPClab_Package.utilitaire_load.basic_load import LabviewFilesReward, LabviewFilesTrajectory, LoadData
from py_NPClab_Package.utilitaire_traitement.TrajectoryTraitement import BasicTraitmentTrajectory
from py_NPClab_Package.traitement_labview.Labview_traitment import AnalyseFromLabview

from py_NPClab_Package.Exemple.CreationSegment import SegmentInit
from py_NPClab_Package.Exemple.CreationEvent import EventInit
from py_NPClab_Package.Exemple.TraitementNeurone import NeuroneInit

dir_profile_pattern: str = r'Y:\Analyse_maxime\profile_pattern'

# dir_save: str = r'Y:\Analyse_maxime\cplx25\save'
# dir_data: str = r'Y:\Analyse_maxime\cplx25'
# dir_spikefile: str = r'Y:\Analyse_maxime\cplx25\clustering\*.txt'
# dir_txt_traj: str = r'Y:\Analyse_maxime\cplx25\fichier_traj\*.txt'

num_segment = 1
name_neurone: str = 'segment1_neurone0'

dir_global = r'Y:\Analyse_maxime'

# ----- det
dir_save: str = r'Y:\Analyse_maxime\cplx07 + bsl\save'
dir_data: str = r'Y:\Analyse_maxime\cplx07 + bsl'
dir_spikefile: str = r'Y:\Analyse_maxime\cplx07 + bsl\clustering\*.txt'
dir_txt_traj: str = r'Y:\Analyse_maxime\cplx07 + bsl\fichier_traj\*.txt'

# ------------------------------------initialisation et creation du set de données ---------------------------

# event = EventInit(dir_data=dir_data)
# event.set_event()
#
# segment = SegmentInit(dir_data=dir_data, dir_save=dir_save)
# segment.set_segment()
#
# neurones = NeuroneInit(dir_data=dir_data, dir_save=dir_save, dir_spikefile=dir_spikefile)
# neurones.set_neurone(neurones=['neurone0_brute'], num_segment=0)
# #
# neurones.set_neurone(neurones=['neurone0_brute', 'neurone1_brute', 'neurone2_brute', 'neurone3_brute'], num_segment=0)

#------------------------------------- parti import data labview ---------------------------------------------------


# chargement de la trajectoire contient "x, y, point"
trajectoire = LoadData.init_data(LabviewFilesTrajectory, dir_txt_traj)
# chargement des temps en ms "temps" contenu dans le fichier rewards
reward = LoadData.init_data(LabviewFilesReward, dir_txt_traj)
print(reward.name_file)
print(trajectoire.name_file)


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


# name_event = 'all_event'
all_event = LoadData.init_data(EventFilesSerialiser, dir_save, 'all_event')

reward = EventRewardNeuralynx()

profile_pattern_synchro = reward.get_profile(dir_pattern=dir_profile_pattern, name_pattern='synchro_video_classic')
profile_pattern_stim = reward.get_profile(dir_pattern=dir_profile_pattern, name_pattern='stimulation_classic')


start_stim, start_stim_index = reward.set_reward(reward_time=all_event.data[f'num segment : {num_segment}']['time'],
                                                 reward_index=all_event.data[f'num segment : {num_segment}']['index'],
                                                 profile_pattern_synchro=profile_pattern_synchro,
                                                 profile_pattern_stim=profile_pattern_stim)
start_stim_index = start_stim_index.astype(dtype=int)


# ------------------------------------- parti reload specification segment ----------------------------------
from py_NPClab_Package.utilitaire_load.basic_load import SegmentBTFilesSerialiser

# name = 'segment_infos'
# num_segment = 0
base_time_segment = LoadData.init_data(SegmentBTFilesSerialiser, dir_save, 'segment_infos', num_segment)

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

# -------------------------------------------- partie reload specifique neurone ---------------------------------
from py_NPClab_Package.utilitaire_load.basic_load import NeuroneFilesSerialiser

# name_neurone = 'segment0_neurone0'
neurone = LoadData.init_data(NeuroneFilesSerialiser, dir_save, name_neurone)

# ------------------------------------- parti preparation des données pour le plot avec spike---------
from py_NPClab_Package.utilitaire_plot.BasicPlotSpike import GenericPlotV2
from py_NPClab_Package.utilitaire_load.basic_load import ImportNeuralynx

raw_brute = LoadData.init_data(ImportNeuralynx, dir_data, 'csc')

plot = GenericPlotV2()
# -----------------------------------------------------------------------------------------------
reward_spike_time = pd.Series(base_time_segment.data[start_stim_index])

from py_NPClab_Package.traitement_global.TraitementNeuroneGlobal import SaveRasterData, ConstructRaster

data_raster = ConstructRaster()
all_data_reward = data_raster.set_data_raster(spike_time=np.array(neurone.data['time']), time_event=reward_spike_time)
reward_plot_data, re_struct = plot.plot_raster_event_spike(raster_spike_time=all_data_reward, name=f'{dir_data}\\reward_{name_neurone}')

c = data_raster.normalisation_raster(re_struct=re_struct, length_neurone=len(neurone.data['time']), data=reward_plot_data)
vecpoint_reward, logkde_reward = plot.plot_kernel_density(c, f'{dir_data}\\reward_{name_neurone}')


# ---------------------------------
all_data_omission = data_raster.set_data_raster(spike_time=np.array(neurone.data['time']), time_event=omission_spike_time)

omission_plot_data, omi_struct = plot.plot_raster_event_spike(raster_spike_time=all_data_omission,
                                                                                   name=f'{dir_data}\\omission_{name_neurone}')
d = data_raster.normalisation_raster(re_struct=omi_struct, length_neurone=len(neurone.data['time']), data=omission_plot_data)

vecpoint_omission, logkde_omission = plot.plot_kernel_density(d, f'{dir_data}\\omission_{name_neurone}')
# -----------------------------------------------

data_save_raster = {}

data_save_raster['reward_'+dir_data.split('\\')[-1]] = [reward_plot_data, all_data_reward, vecpoint_reward, logkde_reward]
data_save_raster['omission_'+dir_data.split('\\')[-1]] = [omission_plot_data, all_data_omission, vecpoint_omission, logkde_omission]

save_data = SaveRasterData()
save_data.save_raster(name_data=name_neurone+dir_data.split('\\')[-1], dir_save=dir_global, data=data_save_raster)
#

# -----------------------------------------

plot.plot_frequence_glissante(neurones=[np.array(neurone.data['time'])],
                              name_neurone=['neurone'],
                              taille_fenetre=15, pas_de_gliss=5, name=name_neurone)
plot.plot_burst_glissant(neurones_times=[neurone.data['time']], neurones_isi=[neurone.data['isi']],
                              name_neurone=['neurone'],
                              taille_fenetre=15, pas_de_gliss=5, name=name_neurone)

plot.plotcorrelogram(neurones=[neurone.data['time']], lag_max=0.5, lenght_of_bin=0.01, name=name_neurone)
#
# intervalle = [1 * 32000, 18 * 32000]  # intervalle par seconde

# plot.plot_neurone_in_raw(intervalle=intervalle_omission, neurones_index_val=[neurone_spike_omission_bool['time_index_in raw'].dropna().astype(dtype=int)], raw_signal=raw_brute.signal_segments['num segment : 1'],
#                          event=all_event, option_other_trigger=omission_recallage.event_index_in_raw,
#                          name='raw plot', option_csc=['CSC2','CSC6','CSC7','CSC10'])

# plot.plot_neurone_in_raw(intervalle=intervalle, neurones_index_val=[neurone_spike_bool['time_index_in raw'].dropna().astype(dtype=int)],
#                          raw_signal=raw_brute.signal_segments['num segment : 0'],
#                          event=all_event['num segment : 0'], name='raw plot', option_csc=['CSC2', 'CSC6', 'CSC7', 'CSC8'])
#
# plot.plot_waveform_neurone(intervalle=intervalle, neurones_index_val=[neurone_spike_bool['time_index_in raw'].dropna().astype(dtype=int)],
#                            raw_signal=raw_brute.signal_segments['num segment : 1'],
#                          event=all_event,
#                          name='raw plot', option_csc=['CSC2','CSC6', 'CSC7', 'CSC8'])