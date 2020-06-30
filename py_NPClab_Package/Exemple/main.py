# -*-coding:utf-8 -*-
import numpy as np
import pandas as pd
from NPClab_Package.traitement_deeplabcut.DeepLabCut_traitment import PreFormDLC
from NPClab_Package.utilitaire_load import LabviewFilesReward, LabviewFilesTrajectory, LoadData, DeepLabCutFileImport
from utilitaire_traitement.TrajectoryTraitement import BasicTraitmentTrajectory
from traitement_labview.Labview_traitment import AnalyseFromLabview

# _______________________________a utiliser pour les data deeplabcut
dir_DLC = r'/data/trajectoire_tarek_deeplapcut/DLC_data'
DLC_brute = LoadData.init_data(DeepLabCutFileImport, dir_DLC,
                               'Chichi_3_cplx_04bis_500mV_03042020-1627DLC_resnet50_testgpuMay2shuffle1_50000.csv')

data_DLC = PreFormDLC()
data_tracking_DLC = data_DLC.load_data_from_DLC(data_brute=DLC_brute.data)

traitment_DLC = BasicTraitmentTrajectory()
data_tracking_brute_DLC = traitment_DLC.data_brute(data_DLC, data_tracking_DLC)
# data_tracking_correc_DLC = traitment_DLC.correction(data_DLC, data_tracking_DLC)
#
# norme_vecteur_entre_point = traitment_DLC.norme_vecteur_entre_points(data_tracking_DLC, name_pointsA='hang',
#                                                                      name_pointsB='hand')

# name = f'\\testsave.csv'
# csv_save = SavingMethodes()
# csv_save.save_csv(path=dir_DLC + name, data=data_tracking_DLC)

# ------------------------------------- parti plot ----------------------------------------
from utilitaire_plot.TrajectoireBasicPlot import SpecifiquePlot
plotcomportement = SpecifiquePlot()
# plotcomportement.plot_norme_vecteur(traitment_DLC.norme_vecteur, data_DLC)
plotcomportement.plot_traj(data_tracking_brute_DLC, data_DLC.items, 'DLC')

#------------------------------------- parti import data labview ---------------------------------------------------

dir_txt_traj = r'D:\Dropbox\python\import_neuralynxv2\data\cplx07 + bsl\fichier traj\*.txt'

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
from utilitaire_traitement.PreFormatData import PreFormatData

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
from NPClab_Package.utilitaire_load import ImportNeuralynx, LoadData
from NPClab_Package.traitement_event import EventFileNeuralynx, EventRewardNeuralynx


dir_data: str = r'/data/cplx07 + bsl'

event_brute = LoadData.init_data(ImportNeuralynx, dir_data, 'event')

Event = EventFileNeuralynx(dir_data=dir_data, event=event_brute)
info_event, all_event = Event.set_event()

reward = EventRewardNeuralynx()

start_stim, start_stim_index = reward.set_reward(reward_time=all_event['num segment : 1']['time'],
                                                 reward_index=all_event['num segment : 1']['index'])
start_stim_index = start_stim_index.astype(dtype=int)

# ------------------------------------ parti spike -----------------------------------------
from traitement_spike import Segment, Spike, PreFormatSpike
from NPClab_Package.utilitaire_load import NeuralynxFilesSpike

dir_spikefile: str = r'D:\Dropbox\python\import_neuralynxv2\data\cplx07 + bsl\clustering\*.txt'

spikefiles = LoadData.init_data(NeuralynxFilesSpike, dir_spikefile)



segment_infos = Segment(dir_data=dir_data)
segment_data = segment_infos.set_segment(event_brute=event_brute.Event, event_final=all_event)

e = PreFormatSpike(dir_data=dir_data, spike_files=spikefiles, segment_infos=segment_data, all_event=all_event)
e.set_spike()

spike = Spike(dir_data=dir_data, name_neurone='neurone0_brute', segment_infos=segment_data, num_segment=0)
spike.set_neurone()
spike = Spike(dir_data=dir_data, name_neurone='neurone0_brute', segment_infos=segment_data, num_segment=1)
spike.set_neurone()
spike = Spike(dir_data=dir_data, name_neurone='neurone1_brute', segment_infos=segment_data, num_segment=0)
spike.set_neurone()
spike = Spike(dir_data=dir_data, name_neurone='neurone1_brute', segment_infos=segment_data, num_segment=1)
spike.set_neurone()

del spike, segment_infos, segment_data, e


# -------------------------------------------- partie reload specifique neurone ---------------------------------
from NPClab_Package.utilitaire_load import NeuroneFilesSerialiser

dir_save = r'/data/cplx07 + bsl/save'
name = 'segment1_neurone0'
neurone = LoadData.init_data(NeuroneFilesSerialiser, dir_save, name)

# ------------------------------------- parti reload specification segment ----------------------------------
from NPClab_Package.utilitaire_load import SegmentFilesSerialiser

dir_save = r'/data/cplx07 + bsl/save'
name = 'segment_infos'
num_segment = 1
base_time_segment = LoadData.init_data(SegmentFilesSerialiser, dir_save, name, num_segment)

# -------------------------------------- parti recallage omission sur temps spike --------------------------
from NPClab_Package.traitement_event import EventFromOther

omission_recallage = EventFromOther()
omission_recallage.start(np.array(omission_time), base_time_segment.data, base_time_segment.data[0])
omission_spike_time = pd.Series(base_time_segment.data[omission_recallage.event_time_in_raw])
# ------------------------------------- parti preparation des données pour le plot avec spike---------
from utilitaire_plot.BasicPlotSpike import GenericPlotV2

plot = GenericPlotV2()

omission_plot_data = plot.plot_raster_event_spike(spike_time=np.array(neurone.data['time']),
                                                  time_event=omission_spike_time, name='omission')
plot.plot_kernel_density(omission_plot_data, 'omission')

reward_spike_time = pd.Series(base_time_segment.data[start_stim_index])
reward_plot_data = plot.plot_raster_event_spike(spike_time=np.array(neurone.data['time']), time_event=reward_spike_time,
                                                name='reward')
plot.plot_kernel_density(reward_plot_data, 'reward')
# -------------------------------------------- partie plot spike ----------------------------------------------

print('fin')