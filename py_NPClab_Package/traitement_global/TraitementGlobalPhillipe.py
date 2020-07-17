from py_NPClab_Package.utlilitaire_saving.Saving_traitment import SaveSerialisation
from py_NPClab_Package.utilitaire_load.basic_load import LabviewFilesReward, LabviewFilesTrajectory
from py_NPClab_Package.utilitaire_traitement.TrajectoryTraitement import BasicTraitmentTrajectory
from py_NPClab_Package.traitement_labview.Labview_traitment import AnalyseFromLabview
from py_NPClab_Package.utilitaire_traitement.PreFormatData import PreFormatData
from py_NPClab_Package.traitement_event.EventTraitement import EventRewardNeuralynx
from py_NPClab_Package.utilitaire_load.basic_load import EventFilesSerialiser, LoadData
from py_NPClab_Package.utilitaire_load.basic_load import SegmentBTFilesSerialiser
from py_NPClab_Package.traitement_event.EventTraitement import EventFromOther
from py_NPClab_Package.utilitaire_load.basic_load import NeuroneFilesSerialiser
from py_NPClab_Package.utilitaire_plot.BasicPlotSpike import GenericPlotV2
from py_NPClab_Package.utilitaire_load.basic_load import ImportNeuralynx

import numpy as np

from pandas import DataFrame, Series
import pandas as pd
from numpy.core.multiarray import ndarray
import glob
import os
from os.path import join
import re
from typing import List, Dict
import pathlib
from sklearn.neighbors import KernelDensity



class Session(object):
    def __init__(self):
        pass
        self.tarjectoire = None
        self.reward = None

    def set_group_neurone(self, dir_txt_traj: str):

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

        # all_event = LoadData.init_data(EventFilesSerialiser, dir_save, 'all_event')
        #
        # reward = EventRewardNeuralynx()
        #
        # profile_pattern_synchro = reward.get_profile(dir_pattern=dir_profile_pattern,
        #                                              name_pattern='synchro_video_classic')
        # profile_pattern_stim = reward.get_profile(dir_pattern=dir_profile_pattern, name_pattern='stimulation_classic')
        #
        # start_stim, start_stim_index = reward.set_reward(
        #     reward_time=all_event.data[f'num segment : {num_segment}']['time'],
        #     reward_index=all_event.data[f'num segment : {num_segment}']['index'],
        #     profile_pattern_synchro=profile_pattern_synchro,
        #     profile_pattern_stim=profile_pattern_stim)
        # start_stim_index = start_stim_index.astype(dtype=int)

        # ------------------------------------- parti reload specification segment ----------------------------------

        # name = 'segment_infos'
        # num_segment = 0
        # base_time_segment = LoadData.init_data(SegmentBTFilesSerialiser, dir_save, 'segment_infos', num_segment)

        # -------------------------------------- parti recallage omission sur temps spike --------------------------

        # omission_recallage = EventFromOther()
        # omission_recallage.start(np.array(omission_time), base_time_segment.data, reward.reward_time_ref_rmz)
        # omission_spike_time = pd.Series(base_time_segment.data[omission_recallage.event_index_in_raw])
        #
        #
        # reward_spike_time = pd.Series(base_time_segment.data[start_stim_index])





class PoolSession(object):
    pass

if __name__ == '__main__':
    dir_txt_traj = r'Y:\py_NPC-Lab_Packages\DataSample\agam3\Equdiff_9\fichier_traj\*.txt'
    exp = Session()
    exp.set_group_neurone(dir_txt_traj=dir_txt_traj)