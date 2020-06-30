from utilitaire_plot.old_version.BasicPlot_separate import *
from traitement_spike.old_version.NeuroneTraitment_threader_separate import LoadSpikeFiles
from NPClab_Package.traitement_event import EventFileNeuralynx, EventRewardNeuralynx
from NPClab_Package.utilitaire_load import LoadData, ImportNeuralynx
import time as chrono


# csc_dir: str = r'D:\Dropbox\python\import_neuralynxv2\data\cplx07 + bsl'
# ext: str = '/*.txt'
# spike_file_folder: str = r'D:\Dropbox\python\import_neuralynxv2\data\cplx07 + bsl\clustering' + ext

dir_neuralynx: str = r'/data/pinp8 06022020'
# csc_dir: str = r'D:\Dropbox\python\import_neuralynxv2\data\cplx07 + bsl'
# csc_dir: str = r'D:\Dropbox\python\import_neuralynxv2\data\equequ1 - test Steve'

ext: str = '/*.txt'
spike_file_folder: str = r'D:\Dropbox\python\import_neuralynxv2\data\pinp8 06022020\clustering' + ext
# dir_txt_traj = r'D:\Dropbox\python\import_neuralynxv2\data\cplx07 + bsl\fichier traj\*.txt'
# dir_neuralynx = r'D:\Dropbox\python\import_neuralynxv2\data\cplx07 + bsl'

spike_files = LoadSpikeFiles(spike_file_folder, dir_neuralynx)
neurones = spike_files.neurone()

t1 = chrono.time()
# dir_txt_traj = r'D:\Dropbox\python\import_neuralynxv2\data\cplx07 + bsl\fichier traj\*.txt'
# dir_neuralynx = r'D:\Dropbox\python\import_neuralynxv2\data\cplx07 + bsl'

# partie utilitaire_neuralynx des données brutes
# trajectoire_file_brute = LoadData.init_data(LabviewFilesTrajectory, dir_txt_traj)
# reward_file_brute = LoadData.init_data(LabviewFilesReward, dir_txt_traj)
event_file_brute = LoadData.init_data(ImportNeuralynx, dir_neuralynx, 'event')

# Crréation de la structure des event du fichier event de neuralynx
Event = EventFileNeuralynx(event=event_file_brute)
info_event, all_event = Event.make_event()
# Extraction des temps des rewards sur la base de temps de l'electrophi
Reward = EventRewardNeuralynx()
# start_stim : contient les temps de reward en sec remis à zéro dans la session
# start_stim_index : contient les index des temps des rewards sur la base du temps electrophi
start_stim, start_stim_index = Reward.make_reward(reward_time=all_event[0]['num segment : 1 time'],
                                                  reward_index=all_event[0]['num segment : 1 index'])

# Extraction des données issu des fichiers produit par labview
# data_AFL = AnalyseFromLabview()
# data_tracking_AFL = data_AFL.load_data_from_labview(trajectoire_file_brute, reward_file_brute)

# data_AFL.plot_event()
# Correction et analyse (vitesse etcs) sur la base de la trajectoire
# traitment_AFL = BasicTraitmentTrajectory()
# data_tracking_AFL = traitment_AFL.correction(data_AFL, data_tracking_AFL)

#
# omission = EventFromOther()
# omission.start(data_AFL.omission, event_file_brute.Event['num segment : 1'].base_time, Reward.reward_time_ref_rmz)

plot = GenericPlot(neurones)
intervalle = [163 * 32000, 164 * 32000]  # intervalle par seconde
# intervalle = [5216000, 5222000] # intervalle par point
#
# plot.plot_neurone_in_raw(intervalle=intervalle, neurone=neurones, event=Event, segment=0, choix_neurone=['neurone0'],
#                          name='test', option_csc=['CSC1', 'CSC2', 'CSC3', 'CSC6'])
plot.plot_burst_glissant(segment=0, taille_fenetre=30, pas_de_gliss=5, name='burst slide')

# raster = plot.plot_raster_event_spike(neurone=neurones, event=Event, segment=0, choix_neurone=['neurone0'], name='neurone0')

# raster = plot.plot_raster_event_spikev2(spike_files= neurones['neurone0'].neurones[1].spike_times_rmz, time_event=, event=Event, segment=0, choix_neurone=['neurone0'], name='neurone0')


# plot.plot_kernel_density(_tmp=raster, name='neurone0')
t2 = chrono.time()

print(f'temps global : {t2 - t1}')

# =================================================
"""
pour save tes temps de spike :
- lorque tu as lancé le code tu reviens dans le terminal python et tu utilise cette ligne
"""
# SavingMethodes.save_spike_text(neurones['neurone0'].neurones[0].spike_times_rmz, 'prout', r'D:\'')
