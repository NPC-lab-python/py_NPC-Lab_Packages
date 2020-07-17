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


class ConstructRaster(SaveSerialisation):

    def __init__(self):
        super().__init__()

    def set_data_raster(self, spike_time: ndarray, time_event: Series, windows: int = 2, before_trig: float = 2) -> DataFrame:
        """
        Cette méthode permet de construire un raster.
        Elle va retourner un dataframe ayant pour columns chaque spike et chaque trig en index

        :param spike_time: les temps des spikes
        :param time_event: les temps des events
        """

        time_spike_try = pd.DataFrame()
        for id, val in enumerate(time_event):
            # _spike_around_norma = (2*(e.iloc[idx][e.iloc[idx].notna()]-(time_event[idx] - fenetre))/((time_event[idx] + fenetre)-(time_event[idx] - fenetre)))-1
            time_spike_try = time_spike_try.append(pd.Series(
                spike_time[((spike_time < time_event[id] + windows) & (spike_time > time_event[id]-before_trig))], name=str(id)) - time_event[id])
        return time_spike_try


    def ksdensity_normaliser(self, time_spike_in_windows, re_struct: Series, length_neurone: int, dur_session: float,
                  fenetre_before: float = 2, fenetre_after: float = 2):
        """

        :param time_spike_in_windows: les temps des spikes
        :param re_struct: correspond au nombre de spike par essaie
        :param length_neurone: nombre de spike
        :param dur_session: durée de la session en seconde
        :param fenetre_before: les temps des events
        :param fenetre_after: les temps des events

        """

        if len(time_spike_in_windows) <= 1:
            Vecvalues = np.array([], dtype=float)
            logkde = np.array([], dtype=float)
            return Vecvalues, logkde
        else:

            Vecvalues = time_spike_in_windows[:, None]
            Vecpoints = np.linspace(-fenetre_before - 0.2, fenetre_after + 0.2, 100)[:, None]
            kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(Vecvalues)
            logkde = kde.score_samples(Vecpoints)

            nb_total_spike_in_session = re_struct.sum()
            nb_try = len(re_struct)
            freq_mean_windows = nb_total_spike_in_session / nb_try

            mean_freq = length_neurone/dur_session
            mean_all = freq_mean_windows / mean_freq

            ksdensity_data = np.exp(logkde) * mean_all
            return Vecpoints, ksdensity_data


    def global_load(self, path: str):

        path_save = f'{path}\\*.dat'
        path_files = [i for i in glob.glob(path_save) if i.find(f'.dat')]
        # e  = pd.Series([],dtype=float)
        reward = pd.DataFrame()
        omission = pd.DataFrame()

        reward_density = pd.DataFrame()
        omission_density = pd.DataFrame()
        vecpoints = pd.DataFrame()


        for i in path_files:
            data = self.load_data_serializer(path=path, name=i.split('\\')[-1].split('.dat')[0])
            # vecpoints = data[list(data.keys())[0]][2]
            e = []
            for i in data.keys():
                # vecpoints=data[i][2]
                if i.find('reward') == 0:
                    print(len(data[i][1].columns))
                    reward = reward.append(data[i][1])
                    reward_density = reward_density.append(pd.Series(data[i][3], name=i.split('\\')[-1].split('.dat')[0]))
                    e.extend([i for i in data[i][2]])
                    vecpoints = vecpoints.append(pd.Series(e, name=i.split('\\')[-1].split('.dat')[0]))

                else:
                    print(len(data[i][1].columns))
                    omission = omission.append(data[i][1])
                    omission_density = omission_density.append(pd.Series(data[i][3], name=i.split('\\')[-1].split('.dat')[0]))


        return reward_density, omission_density, vecpoints


class SearchFiles(object):
    """

    """
    def search(self, path: str, save_folde: str = 'save', file_type: str = '.dat', num_segment: int = 0, word_init: str = 'segment'):

        pattern_name = f'{word_init}{num_segment}_neurone'
        pattern = f'{path}\**\{save_folde}\*{file_type}'
        list_files: List[str] = []
        list_files = self._search_files_(list_path=glob.glob(pattern, recursive=True), pattern_name=pattern_name, list_files=list_files)
        return list_files

    def search_labview_data(self, path: str, file_contain_word: str, search_folder: str = 'fichier_traj', file_type: str = '.txt'):

        pattern_name = f'{file_contain_word}'
        pattern = f'{path}\**\{search_folder}\*{file_type}'
        list_files: List[str] = []
        list_files = self._search_files_(list_path=glob.glob(pattern, recursive=True), pattern_name=pattern_name,
                                       list_files=list_files)
        return list_files

    def _search_files_(self, list_path, pattern_name, list_files: List[str]) -> List[str]:
        if len(list_path) == 0:
            return list_files
        else:
            # resul = re.search(pattern=pattern_name, string=list_path[0])

            resul = re.search(pattern=pattern_name, string=list_path[0].split('\\')[-1])
            if isinstance(resul, type(None)):
                pass
            else:
                list_files.append(list_path[0])
            return self._search_files_(list_path=list_path[1:], pattern_name=pattern_name, list_files=list_files)


class SaveRasterData(SaveSerialisation):
    def __init__(self):
        super().__init__()

    def save_raster(self, name_data: str, dir_save: str, data: DataFrame):

        self._set_conf_(name=name_data, dir_save_conf=dir_save, data=data)

class GlobalTraitement(object):
    def set_list_dir_group(self, list_files):
        for val in list_files:
            base_path = val.split('\\')
            base_name_neurone = val.split('\\')[-1].split('.')[0]
            base_num_segment = int(val.split('\\')[-1].split('_')[0][-1])
            e = val.split('\\save')
            dir_profile_pattern: str = f'{base_path[0]}\{base_path[1]}\profile_pattern'
            dir_save: str = f'{e[0]}\save'
            dir_data: str = f'{e[0]}'
            dir_spikefile: str = f'{e[0]}\clustering\*.txt'
            dir_txt_traj: str = f'{e[0]}\{"fichier_traj"}\*.txt'


            num_segment = base_num_segment
            name_neurone: str = base_name_neurone
            if val.find('Det') == -1:
                dir_global = f'{base_path[0]}\{base_path[1]}\global_raster'
            else:
                dir_global = f'{base_path[0]}\{base_path[1]}\global_raster_det'

            self.set_group_neurone(dir_save=dir_save, dir_data=dir_data, dir_global=dir_global, dir_spikefile=dir_spikefile,
                                   dir_profile_pattern=dir_profile_pattern, dir_txt_traj=dir_txt_traj,
                                   name_neurone=name_neurone, num_segment=num_segment)



    def set_group_neurone(self, dir_data, dir_txt_traj, dir_save, dir_global, dir_spikefile: str,
                          dir_profile_pattern: str, num_segment, name_neurone):

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


        # name_event = 'all_event'
        all_event = LoadData.init_data(EventFilesSerialiser, dir_save, 'all_event')

        reward = EventRewardNeuralynx()

        profile_pattern_synchro = reward.get_profile(dir_pattern=dir_profile_pattern,
                                                     name_pattern='synchro_video_classic')
        profile_pattern_stim = reward.get_profile(dir_pattern=dir_profile_pattern, name_pattern='stimulation_classic')

        start_stim, start_stim_index = reward.set_reward(reward_time=all_event.data[f'num segment : {num_segment}']['time'],
                                                         reward_index=all_event.data[f'num segment : {num_segment}']['index'],
                                                         profile_pattern_synchro=profile_pattern_synchro,
                                                         profile_pattern_stim=profile_pattern_stim)
        start_stim_index = start_stim_index.astype(dtype=int)

        # ------------------------------------- parti reload specification segment ----------------------------------

        # name = 'segment_infos'
        # num_segment = 0
        base_time_segment = LoadData.init_data(SegmentBTFilesSerialiser, dir_save, 'segment_infos', num_segment)

        # -------------------------------------- parti recallage omission sur temps spike --------------------------

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

        # name_neurone = 'segment0_neurone0'
        neurone = LoadData.init_data(NeuroneFilesSerialiser, dir_save, name_neurone)

        # ------------------------------------- parti preparation des données pour le plot avec spike---------


        raw_brute = LoadData.init_data(ImportNeuralynx, dir_data, 'csc')

        plot = GenericPlotV2()
        # -----------------------------------------------------------------------------------------------
        reward_spike_time = pd.Series(base_time_segment.data[start_stim_index])


        data_raster = ConstructRaster()
        all_data_reward = data_raster.set_data_raster(spike_time=np.array(neurone.data['time']),
                                                      time_event=reward_spike_time)
        reward_plot_data, re_struct = plot.plot_raster_event_spike(raster_spike_time=all_data_reward,
                                                                   name=f'{dir_data}\\reward_{name_neurone}')

        dur_session = all_event.data[f'num segment : {num_segment}']['time'].iloc[-1]-reward.reward_time_ref_rmz

        vecpoint_reward, kde_reward = data_raster.ksdensity_normaliser(time_spike_in_windows=reward_plot_data, re_struct=re_struct,
                                             length_neurone=len(neurone.data['time']),
                                             dur_session=dur_session)
        plot.plot_kernel_density(Vecpoints=vecpoint_reward, ksdensity_normaliser=kde_reward,
                                 name=f'{dir_data}\\reward_{name_neurone}')

        # ---------------------------------
        all_data_omission = data_raster.set_data_raster(spike_time=np.array(neurone.data['time']),
                                                        time_event=omission_spike_time)

        omission_plot_data, omi_struct = plot.plot_raster_event_spike(raster_spike_time=all_data_omission,
                                                                      name=f'{dir_data}\\omission_{name_neurone}')
        vecpoint_omission, kde_omission = data_raster.ksdensity_normaliser(time_spike_in_windows=omission_plot_data, re_struct=omi_struct,
                                             length_neurone=len(neurone.data['time']),
                                             dur_session=dur_session)

        plot.plot_kernel_density(Vecpoints=vecpoint_omission, ksdensity_normaliser=kde_omission,
                                 name=f'{dir_data}\\omission_{name_neurone}')
        # -----------------------------------------------

        data_save_raster = {}

        data_save_raster['reward_' + dir_data.split('\\')[-1]] = [reward_plot_data, all_data_reward, vecpoint_reward,
                                                                  kde_reward]
        data_save_raster['omission_' + dir_data.split('\\')[-1]] = [omission_plot_data, all_data_omission,
                                                                    vecpoint_omission, kde_omission]

        save_data = SaveRasterData()
        save_data.save_raster(name_data=name_neurone + dir_data.split('\\')[-1], dir_save=dir_global,
                              data=data_save_raster)

        # plot.plot_frequence_glissante(neurones=[np.array(neurone.data['time'])],
        #                               name_neurone=['neurone'],
        #                               taille_fenetre=15, pas_de_gliss=5, name=f'{dir_data}\\freq_gli_{name_neurone}')
        #
        # plot.plot_burst_glissant(neurones_times=[neurone.data['time']], neurones_isi=[neurone.data['isi']],
        #                          name_neurone=['neurone'],
        #                          taille_fenetre=15, pas_de_gliss=5, name=f'{dir_data}\\burst_gli_{name_neurone}')
        #
        # plot.plotcorrelogram(neurones=[neurone.data['time']], lag_max=0.5, lenght_of_bin=0.01, name=f'{dir_data}\\correlogra_{name_neurone}')




if __name__ == '__main__':
    files = SearchFiles()
    # list_files = files.search(r'Y:\Analyse_maxime')
    list_files_reward = files.search_labview_data(path=r'Y:\Analyse_maxime', file_contain_word='reward')
    list_files_traj = files.search_labview_data(path=r'Y:\Analyse_maxime', file_contain_word='traj')
    list_files_comportment = pd.DataFrame({'files_trajectory': list_files_traj, 'files_reward': list_files_reward})

    traitement = GlobalTraitement()
    traitement.set_list_dir_group(list_files=list_files)