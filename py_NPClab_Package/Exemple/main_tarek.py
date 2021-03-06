# -*-coding:utf-8 -*-
import numpy as np
import pandas as pd
from NPClab_Package.utilitaire_load.basic_load import LabviewFilesReward, LabviewFilesTrajectory, LoadData
from NPClab_Package.utilitaire_traitement.TrajectoryTraitement import BasicTraitmentTrajectory
from NPClab_Package.traitement_labview.Labview_traitment import AnalyseFromLabview

# _______________________________a utiliser pour les data deeplabcut
# dir_DLC = r'D:\Dropbox\python\import_neuralynxv2\data\trajectoire_tarek_deeplapcut\DLC_data'
# DLC_brute = LoadData.init_data(DeepLabCutFileImport, dir_DLC,
#                                'Chichi_3_cplx_04bis_500mV_03042020-1627DLC_resnet50_testgpuMay2shuffle1_50000.csv')
#
# data_DLC = PreFormDLC()
# data_tracking_DLC = data_DLC.load_data_from_DLC(data_brute=DLC_brute.data)
#
# traitment_DLC = BasicTraitmentTrajectory()
# data_tracking_brute_DLC = traitment_DLC.data_brute(data_DLC, data_tracking_DLC)
# # data_tracking_correc_DLC = traitment_DLC.correction(data_DLC, data_tracking_DLC)
# #
# # norme_vecteur_entre_point = traitment_DLC.norme_vecteur_entre_points(data_tracking_DLC, name_pointsA='hang',
# #                                                                      name_pointsB='hand')
#
# # name = f'\\testsave.csv'
# # csv_save = SavingMethodes()
# # csv_save.save_csv(path=dir_DLC + name, data=data_tracking_DLC)
#
# # ------------------------------------- parti plot ----------------------------------------
# from utilitaire_plot.TrajectoireBasicPlot import SpecifiquePlot
# plotcomportement = SpecifiquePlot()
# # plotcomportement.plot_norme_vecteur(traitment_DLC.norme_vecteur, data_DLC)
# plotcomportement.plot_traj(data_tracking_brute_DLC, data_DLC.items, 'DLC')

#------------------------------------- parti import data labview ---------------------------------------------------

dir_txt_traj = r'Y:\python\import_neuralynxv2\data\cplx07 + bsl\fichier traj\*.txt'

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
from NPClab_Package.utilitaire_traitement.PreFormatData import PreFormatData

data_formater = PreFormatData()

omission_time = pd.Series(data_AFL.omission, name='omission')
# ---------------------------------------------
data_formater.make_event_around(data=traitment_AFL.norme_vecteur, reward=omission_time)

# ------------------------------------- parti plot ----------------------------------------
# from utilitaire_plot.TrajectoireBasicPlot import SpecifiquePlot
# plotcomportement = SpecifiquePlot()
# plotcomportement.plot_event_around(['trajectoire'], data_formater.around_event)
# plotcomportement.plot_norme_vecteur(traitment_AFL.norme_vecteur, data_AFL)
# plotcomportement.plot_traj(data_tracking_AFL, data_AFL.couple_de_points[0])

# -------------------------------------------- partie event ---------------------------------------------------
from NPClab_Package.utilitaire_load.basic_load import ImportNeuralynx, LoadData
from NPClab_Package.traitement_event.EventTraitement import EventFileNeuralynx, EventRewardNeuralynx


dir_data: str = r'Y:\python\import_neuralynxv2\data\cplx07 + bsl'

event_brute = LoadData.init_data(ImportNeuralynx, dir_data, 'event')

Event = EventFileNeuralynx(dir_data=dir_data, event=event_brute)
info_event, all_event = Event.set_event()

reward = EventRewardNeuralynx()

start_stim, start_stim_index = reward.set_reward(reward_time=all_event['num segment : 1']['time'],
                                                 reward_index=all_event['num segment : 1']['index'])
start_stim_index = start_stim_index.astype(dtype=int)

# ------------------------------------ parti spike -----------------------------------------
from NPClab_Package.traitement_spike.NeuroneTraitment import Segment, Spike, PreFormatSpike, CleaningSpikeTime
from NPClab_Package.utilitaire_load.basic_load import NeuralynxFilesSpike

dir_spikefile: str = r'Y:\python\import_neuralynxv2\data\cplx07 + bsl\clustering\*.txt'

spikefiles = LoadData.init_data(NeuralynxFilesSpike, dir_spikefile)



segment_infos = Segment(dir_data=dir_data)
segment_data = segment_infos.set_segment(event_brute=event_brute.Event, event_final=all_event)

e = PreFormatSpike(dir_data=dir_data, spike_files=spikefiles, segment_infos=segment_data, all_event=all_event)
e.set_spike()

spike = Spike(dir_data=dir_data, name_neurone='neurone0_brute', segment_infos=segment_data, num_segment=0)
spike.set_neurone()
spike = Spike(dir_data=dir_data, name_neurone='neurone0_brute', segment_infos=segment_data, num_segment=1)
spike.set_neurone()
# spike = Spike(dir_data=dir_data, name_neurone='neurone1_brute', segment_infos=segment_data, num_segment=0)
# spike.set_neurone()
# spike = Spike(dir_data=dir_data, name_neurone='neurone1_brute', segment_infos=segment_data, num_segment=1)
# spike.set_neurone()

del spike, segment_infos, segment_data, e


# -------------------------------------------- partie reload specifique neurone ---------------------------------
from NPClab_Package.utilitaire_load.basic_load import NeuroneFilesSerialiser

dir_save = r'Y:\python\import_neuralynxv2\data\cplx07 + bsl\save'
name = 'segment1_neurone0'
neurone = LoadData.init_data(NeuroneFilesSerialiser, dir_save, name)


# ------------------------------------- parti reload specification segment ----------------------------------
from NPClab_Package.utilitaire_load.basic_load import SegmentFilesSerialiser

dir_save = r'Y:\python\import_neuralynxv2\data\cplx07 + bsl\save'
name = 'segment_infos'
num_segment = 1
base_time_segment = LoadData.init_data(SegmentFilesSerialiser, dir_save, name, num_segment)

# -------------------------------------- parti recallage omission sur temps spike --------------------------
from NPClab_Package.traitement_event.EventTraitement import EventFromOther

omission_recallage = EventFromOther()
omission_recallage.start(np.array(omission_time), base_time_segment.data, reward.reward_time_ref_rmz)
omission_spike_time = pd.Series(base_time_segment.data[omission_recallage.event_index_in_raw])

# ----------------------------- nettoyage du neurone autour de la stim
dir_data = r'Y:\python\import_neuralynxv2\data\cplx07 + bsl'
new_spike = CleaningSpikeTime(dir_data=dir_data, name_neurone='segment1_neurone0',
                              start_stim=start_stim, time_ref_synchro=reward.reward_time_ref_rmz)
neurone_spike_bool = new_spike.load_neurone()

# new_omission = CleaningSpikeTime(dir_data=dir_data, name_neurone='segment1_neurone0',
#                               start_stim=omission_spike_time, time_ref_synchro=reward.reward_time_ref_rmz)
# neurone_spike_omission_bool = new_omission.load_other()

# ------------------------------------- parti preparation des données pour le plot avec spike---------
from NPClab_Package.utilitaire_plot.BasicPlotSpike import GenericPlotV2

raw_brute = LoadData.init_data(ImportNeuralynx, dir_data, 'csc')

plot = GenericPlotV2()

omission_plot_data = plot.plot_raster_event_spike(spike_time=np.array(neurone_spike_bool['time'].dropna()),
                                                  time_event=omission_spike_time, name='omission')
plot.plot_kernel_density(omission_plot_data, 'omission')

reward_spike_time = pd.Series(base_time_segment.data[start_stim_index])
reward_plot_data = plot.plot_raster_event_spike(spike_time=np.array(neurone_spike_bool['time'].dropna()),
                                                time_event=reward_spike_time,
                                                name='reward')
plot.plot_kernel_density(reward_plot_data, 'reward')
# -------------------------------------------- partie plot spike ----------------------------------------------
from NPClab_Package.utilitaire_load.basic_load import NeuroneFilesSerialiser

# dir_save = r'Y:\python\import_neuralynxv2\data\cplx07 + bsl\save'
# name = 'segment1_neurone0'
# neurone0 = LoadData.init_data(NeuroneFilesSerialiser, dir_save, name)

plot.plot_frequence_glissante(neurones=[np.array(neurone_spike_bool['time'].dropna())],
                              name_neurone=['neurone'],
                              taille_fenetre=15, pas_de_gliss=5, name='neuron')
plot.plot_burst_glissant(neurones_times=[neurone_spike_bool['time'].dropna()], neurones_isi=[neurone_spike_bool['isi'].dropna()],
                              name_neurone=['neurone'],
                              taille_fenetre=15, pas_de_gliss=5, name='neuron')

plot.plotcorrelogram(neurones=[np.array(neurone_spike_bool['time'].dropna())], lag_max=0.5, lenght_of_bin=0.001, name='cross')

intervalle = [1 * 32000, 18 * 32000]  # intervalle par seconde
intervalle_omission = [omission_recallage.event_index_in_raw[0]-32000, omission_recallage.event_index_in_raw[1]+(32000*8)]  # intervalle par seconde

# plot.plot_neurone_in_raw(intervalle=intervalle_omission, neurones_index_val=[neurone_spike_omission_bool['time_index_in raw'].dropna().astype(dtype=int)], raw_signal=raw_brute.signal_segments['num segment : 1'],
#                          event=all_event, option_other_trigger=omission_recallage.event_index_in_raw,
#                          name='raw plot', option_csc=['CSC2','CSC6','CSC7','CSC10'])

plot.plot_neurone_in_raw(intervalle=intervalle_omission, neurones_index_val=[neurone_spike_bool['time_index_in raw'].dropna().astype(dtype=int)], raw_signal=raw_brute.signal_segments['num segment : 1'],
                         event=all_event, option_other_trigger=omission_recallage.event_index_in_raw,
                         name='raw plot', option_csc=['CSC2', 'CSC6', 'CSC7', 'CSC10'])

plot.plot_waveform_neurone(intervalle=intervalle, neurones_index_val=[neurone_spike_bool['time_index_in raw'].dropna().astype(dtype=int)], raw_signal=raw_brute.signal_segments['num segment : 1'],
                         event=all_event,
                         name='raw plot', option_csc=['CSC2','CSC6','CSC7','CSC10'])