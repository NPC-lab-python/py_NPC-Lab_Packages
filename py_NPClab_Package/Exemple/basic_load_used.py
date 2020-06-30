from NPClab_Package.utilitaire_load.basic_load import LoadData, NeuralynxFilesSpike, LabviewFilesReward, LabviewFilesTrajectory, NeuroneFilesSerialiser, SegmentFilesSerialiser, DeepLabCutFileImport, ImportNeuralynx

"""
Pour loader un neurone : 
dir_save = r'D:\Dropbox\python\import_neuralynxv2\data\cplx07 + bsl\save'
name = 'segment1_neurone0'
neurone = LoadData.init_data(NeuroneFilesSerialiser, dir_save, name)

"""

# dir_txt = r'/data/cplx07 + bsl/fichier traj/shenron1_cplx_07_filter1_08042020-1527_reward.txt'
# dir_csv = r'D:\Dropbox\python\import_neuralynxv2\data\trajectoire_tarek_deeplapcut\Chichi_3_cplx_04bis_500mV_03042020-1627DLC_mobnet_100_FiberApr1shuffle1_160000.csv'
# dir_txt_traj = r'D:\Dropbox\python\import_neuralynxv2\data\cplx07 + bsl\fichier traj\shenron1_cplx_07_filter1_08042020-1527traj.txt'

# dir_txt_traj = r'D:\Dropbox\python\import_neuralynxv2\data\cplx07 + bsl\fichier traj\*.txt'
# dir_neuralynx = r'D:\Dropbox\python\import_neuralynxv2\data\cplx07 + bsl'
# dir_spikefile = r'D:\Dropbox\python\import_neuralynxv2\data\cplx07 + bsl\clustering\*.txt'
#
# dir_save = r'D:\Dropbox\python\import_neuralynxv2\data\cplx07 + bsl\save'
# name = 'segment1_neurone0'
# neurone = LoadData.init_data(NeuroneFilesSerialiser, dir_save, name)

# trajectoire = LoadData.init_data(LabviewFilesTrajectory, dir_txt_traj)
# reward = LoadData.init_data(LabviewFilesReward, dir_txt_traj)
# Event_brute = LoadData.init_data(ImportNeuralynx, dir_neuralynx, 'event')
# dir_DLC = r'D:\Dropbox\python\import_neuralynxv2\data\trajectoire_tarek_deeplapcut\DLC_data'
#
# DLC_brute = LoadData.init_data(DeepLabCutFileImport, dir_DLC, 'Chichi_3_cplx_04bis_500mV_03042020-1627DLC_resnet50_testgpuMay2shuffle1_50000.csv')

dir_save = r'D:\Dropbox\python\import_neuralynxv2\data\cplx07 + bsl\save'
name = 'segment_infos'
segment_infos_loaded = LoadData.init_data(SegmentFilesSerialiser, dir_save, name, 1)

print('fin')

