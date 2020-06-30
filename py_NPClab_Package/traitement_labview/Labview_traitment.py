import pandas as pd
from pandas import DataFrame, Series, RangeIndex
from itertools import combinations as combi
import re
import itertools as iter
import sys
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure
from matplotlib.axes._axes import Axes
import time as chrono
import numpy as np
from scipy.interpolate import interp1d
from numpy.core.multiarray import ndarray
from typing import List, Dict, Union, Tuple
from NPClab_Package.utilitaire_load.basic_load import LabviewFilesReward,LabviewFilesTrajectory,LoadData
from NPClab_Package.utilitaire_traitement.ExtractionVal import ExtractEvent
from NPClab_Package.utlilitaire_saving.Saving_traitment import SaveSerialisation
import logging
from NPClab_Package.utilitaire_traitement.Decorateur import mesure
from collections import defaultdict

Norme_Coordonnee_Vecteur = Tuple[Series, Series]
Trajectoire = Tuple[ndarray, ndarray, ndarray]
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from abc import ABC, abstractmethod, abstractclassmethod


class PrePlot(ABC):

    @abstractmethod
    def _couple_items_(self, items: List[str]) -> List[List[str]]:
        pass

    @abstractmethod
    def _make_good_format_data_(self, data_brute: DataFrame, option=None) -> DataFrame:
        pass


class AnalyseFromLabview(PrePlot, ExtractEvent):
    """

    """

    sampling_video = 50

    def __init__(self):
        super().__init__()
        self.sample = 50
        self.bin_decoupage = 1000
        self.all_data: DataFrame = None
        self.passage_from_traj: ndarray = None

        self.reward_omission_from_traj: Tuple[Series, Series] = None
        self._reward_raw: ndarray = np.array([], dtype=int)
        self.reward_from_txt: Series = pd.Series(dtype=int)
        self.base_time: Series = pd.Series(dtype=int)
        self.item: List[str] = None
        self.couple_de_points: List[List[str]] = None
        self.items: List[str] = ['trajectoire']

    def _make_good_format_data_(self, data_brute: DataFrame, option=None) -> DataFrame:
        """
        permet de rajouter les series bool des rewards et des omissions
        :param data_brute: all_data (trajectoire x  et y + reward)
        :param option: reward_omission_from_traj: Tuple[Series,Series]
        :return: un dataframe contenant toutes les données par frame
        """
        data_brute[option[0].name] = np.array(option[0])
        data_brute[option[1].name] = np.array(option[1])
        _tmp_items_ = list(data_brute.keys())

        self.item = _tmp_items_
        self.couple_de_points: List[List[str]] = self._couple_items_(_tmp_items_)
        return data_brute

    def _couple_items_(self, items: List[str]) -> List[List[str]]:
        couple = [[items[0], items[1]]]
        return couple

    def _make_basetime_from_traj_(self, trajectoire: Trajectoire) -> Series:
        """
        construction des vecteurs temps pour le plot de la longueur de la trajectoire
        :return:
        """
        longueur_trajectoire: int = len(trajectoire[0])
        time_index = pd.RangeIndex(0, longueur_trajectoire, 1)
        base_time: Series = pd.Series(np.arange(0, longueur_trajectoire * self.sampling_video, self.sampling_video), time_index, dtype=int)
        return base_time

    @mesure
    def _extraction_reward_times_(self, reward_time: Series, base_time: Series) -> Series:
        """
            extraction des temps des rewards recallés sur la base de temps de la trajectoire (video
        :param reward_time: temps contenue dans le fichier
        :param base_time:
        :return:
        """
        logging.debug(f'Nombre de reward from txt : {len(reward_time)}')

        rr = pd.Series([], dtype=int)
        for i in range(len(reward_time)):
            _tmp = base_time[(base_time >= reward_time.iloc[i] - 250) & (base_time <= reward_time.iloc[i] + 250)]
            y = _tmp - reward_time.iloc[i]
            y = y.abs()
            y = y[y == y.min()]
            if len(y) > 1:
                y = y.drop_duplicates()
            rr = rr.append(base_time[y.index])


        # _reward_from_txt = self._extraction_event_times_(event_time=reward_time, base_time=base_time)
        # reward_from_txt: Series = pd.Series(base_time[_reward_from_txt], dtype=int)
        return rr


    def plot_event(self, idx: int, ax: Axes, type_reward: str = None):
        """
        methode à utiliser dans une methode de plot
        permet d'avoir une méthode qui va plot les event sur un autre plot
        :param idx:
        :param ax:
        :param type_reward:
        :return:
        """
        # fig = BasicPlot.crea_fig()
        # items = list(self.all_data.keys())
        # axis_list = BasicPlot.axis_list([items[2]], fig)
        # base_time = self.base_time/1000
        base_time = list(range(0, len(self.base_time)))

        colors = ['g', 'b', 'r', 'g', 'b', 'r', 'g', 'b', 'r']
        # ax: Axes
        # for idx, ax in enumerate(axis_list):
            # ax.plot(self.data[args[idx][0]][171:175], self.data[args[idx][1]][171:175], colors[idx]+'o-')
        if type_reward == 'rewards' or type_reward == 'omissions':
            for i in enumerate(self.all_data.loc[self.all_data[type_reward] == True, type_reward].index):
                ax.plot([base_time[i[1]], base_time[i[1]]],
                        [0, self.all_data.loc[i[1], type_reward]+100], colors[idx]+'o-')
        else:
            for i in enumerate(self.all_data.loc[self.all_data['rewards'] == True, 'rewards'].index):
                ax.plot([base_time[i[1]], base_time[i[1]]],
                        [0, self.all_data.loc[i[1], 'rewards']+100], colors[1]+'o-')
            for i in enumerate(self.all_data.loc[self.all_data['omissions'] == True, 'omissions'].index):
                ax.plot([base_time[i[1]], base_time[i[1]]],
                        [0, self.all_data.loc[i[1], 'omissions']+300], colors[2]+'o-')
        return ax
        #     ax.set(ylim=(0, 500))
        # fig.show()

    def _make_data_from_traj_(self, trajectoire: Trajectoire) -> DataFrame:
        """
        Transformation du fichier xxxtraj.txt en DataFrame
        :param trajectoire: tuple(ndarray)
        :return:
        """
        event_time1: Series = pd.Series(trajectoire[0], name='trajectoire_x')
        event_time2: Series = pd.Series(trajectoire[1], name='trajectoire_y')
        event_time3: Series = pd.Series(trajectoire[2], name='reward')
        data: DataFrame = pd.DataFrame(
            {event_time1.name: event_time1, event_time2.name: event_time2, event_time3.name: event_time3})
        return data

    def make_reward_from_traj(self, data_reward: Series, reward_from_txt: Series):
        self.passage_from_traj = self._search_passage_from_traj_(data_reward, self.base_time)
        self.reward_omission_from_traj = self._search_reward_omission_from_traj_(self.passage_from_traj, reward_from_txt, self.base_time)

    def _search_passage_from_traj_(self, data: Series, base_time: Series) -> Series:
        """
        Extraction de la totalité des passages sur les points. reward et ommission.
        Notes : pour retomber sur l'indexation du fichier "*traj.txt",
                je réindex la series à partir de 1.
                le temps correspond à la nouvelle frame.

        :param data: correspond à la colonne des passage sur les points par frame
        :return: la series nomé passage from traj
        """
        passage = pd.Series(False, index=pd.RangeIndex(0, len(base_time)), name='passage from traj', dtype=bool)
        new_index = list(range(1, len(data) + 1))
        data = pd.Series(np.array(data), index=new_index)
        for idx in range(1, len(data) - 1):
            if data[idx] != data[idx + 1]:
                passage.loc[idx+1] = True
        return passage

    def _search_reward_omission_from_traj_(self, passage_from_traj: Series, reward_from_txt: Series, base_time: Series) -> Tuple[Series, Series]:
        """
        Extraction de la totalité des passages sur les points en ommission.
        création de series de bool pour les rewards et les omissions

        fonctionnement :
            on regarde si pour chaque passage sur un point (donc un temps) il y a un temps correspond +/- le delta_reward_video
            dans le fichier reward

        Il existe un delta entre la video et les temps du fichier reward "delta_reward_video" qui est variable

        :param passage_from_traj:
        :param reward_from_traj: colonne des passage sur les points
        :param reward_from_txt: temps des rewards uniquement

        :return:
        """
        delta_reward_video = 300
        _omission = np.array([], dtype=int)
        _reward = np.array([], dtype=int)

        self.time_passage = self.base_time[passage_from_traj]

        for i in range(len(self.time_passage)):
            _tmp = reward_from_txt[(reward_from_txt >= self.time_passage.iloc[i] - delta_reward_video) & (
                        reward_from_txt <= self.time_passage.iloc[i] + delta_reward_video)]
            if _tmp.empty:
                _omission = np.append(_omission, self.time_passage.index[i])
            else:
                _reward = np.append(_reward, self.time_passage.index[i])


        omission = pd.Series(False, index=pd.RangeIndex(0, len(base_time)), name='omissions', dtype=bool)
        reward = pd.Series(False, index=pd.RangeIndex(0, len(base_time)), name='rewards', dtype=bool)
        omission[_omission] = True
        reward[_reward] = True
        return reward, omission
        # TODO : il y a un vraie problème de précision entre le fichier temps reward et les frames du fichier trajectoire

    @property
    def omission(self) -> Series:
        return pd.Series(self.base_time[self.all_data['omissions']],
                  index=self.all_data['omissions'][self.all_data['omissions'] == True].index)
    @property
    def reward(self) -> Series:
        return pd.Series(self.base_time[self.all_data['rewards']],
                  index=self.all_data['rewards'][self.all_data['rewards'] == True].index)
    @property
    def delta_time_reward(self) -> ndarray:
        return np.array(self.base_time[self.all_data['rewards']]) - np.array(self.reward_from_txt)
    @property
    def format_correction(self):
        format_data = defaultdict(DataFrame)
        format_data['trajectoire'] = pd.DataFrame(
            {'x': self.all_data['trajectoire_x'], 'y': self.all_data['trajectoire_y']})
        return format_data


    def load_data_from_labview(self, trajectoire: LabviewFilesTrajectory, reward: LabviewFilesReward) -> DataFrame:
        """
        param reward: correspond au temps brute directement recupéré du fichier txt issue de LabviewFilesReward
        """
        self.base_time = self._make_basetime_from_traj_(trajectoire.data)
        self.all_data = self._make_data_from_traj_(trajectoire.data)
        self.reward_from_txt = self._extraction_reward_times_(reward.data, self.base_time)
        self.make_reward_from_traj(self.all_data['reward'], self.reward_from_txt)
        self.all_data = self._make_good_format_data_(self.all_data, self.reward_omission_from_traj)
        # SavingMethodes.save_data_text(data=self.omission, name='omission_cpl-07', path= trajectoire.dir.split('*')[0])
        # SavingMethodes.save_data_text(data=self.reward, name='reward_cpl-07', path= reward.dir.split('*')[0])

        ss = self.all_data['rewards'] | self.all_data['omissions']
        rewards = self.all_data['reward'][self.all_data['rewards']]
        omissions = self.all_data['reward'][self.all_data['omissions']]
        ttt = pd.Series(np.nan, index=pd.RangeIndex(0, len(self.all_data['reward'])), name='transitions', dtype=str)
        ttt[rewards.index] = 'reward'
        ttt[omissions.index] = 'omission'
        transition = ttt[ttt.notna()]
        return self.all_data

    def classification_reward(self):
        """
        Le point "-1" est remplacé par "10" précedament
        on obtient un dataframe contenant toutes les transitions entre les points
        """
        # zz = [np.array([str(i[0]) + str(i[1])])[0] for i in iter.permutations(set(self.all_data['reward']), 2)]
        e = self.all_data['reward'].astype(str)
        f = np.array(e)
        f = f.sum()
        model_pattern_transition = ['0011', '001010', '0022', '1100', '111010', '1122', '101000', '01011', '01022', '2200', '2211', '221010']
        transitions_points = pd.DataFrame(columns=model_pattern_transition)
        for i in model_pattern_transition:
            iterator_find = re.finditer(i, f)
            transition = []
            for match in iterator_find:
                transition.append([match.span()[0], match.span()[1]])
            transitions_points[i] = pd.Series(transition)
        self.transitions_points = transitions_points
    # TODO methode à terminer en fonction de comment on va sa servir et voir comment on rentre le model de pattern
# class BasicPlot(object):
#
#     @staticmethod
#     def crea_fig() -> Figure:
#         """
#         Création de la figure
#         :return:
#         """
#         fig = plt.figure(4, figsize=(20, 20))
#         return fig
#
#     @staticmethod
#     def axis_list(items: List[str], fig1: Figure) -> List[Axes]:
#         """
#         Création des axes
#         :param items:
#         :param fig1:-
#         :return:
#         """
#         axis_list: List = []
#         if len(items) != 1:
#             for ind_axsubplot in range(1, len(items) + 1):
#                 axis_list.append(fig1.add_subplot(np.ceil(len(items) / 2), 2, ind_axsubplot))
#         else:
#             axis_list.append(fig1.add_subplot(len(items), 1, 1))
#         return axis_list
#
#
# class SpecifiquePlot(object):
#
#     def plot_traj(self, data: DataFrame, *args):
#
#         fig1 = BasicPlot.crea_fig()
#         items_traj: List[str] = args
#         axis_list = BasicPlot.axis_list(items_traj, fig1)
#         colors = ['g', 'b', 'r', 'g', 'b', 'r', 'g', 'b', 'r']
#         ax: Axes
#         for idx, ax in enumerate(axis_list):
#             # ax.plot(self.data[args[idx][0]][171:175], self.data[args[idx][1]][171:175], colors[idx]+'o-')
#             ax.plot(data[args[idx][0]], data[args[idx][1]], colors[idx]+'o-')
#             ax.set(xlim=(0, 800), ylim=(0, 800))
#         fig1.show()
#
#     def plot_norme_vecteur(self, norme_vecteur: DataFrame, option_event: AnalyseFromLabview = None):
#         try:
#             assert len(norme_vecteur) > 0
#
#             assert isinstance(norme_vecteur, DataFrame)
#
#             fig1 = BasicPlot.crea_fig()
#             items_norme_vecteur: List[str] = list(norme_vecteur.keys())
#             axis_list = BasicPlot.axis_list(items_norme_vecteur, fig1)
#             colors =['g', 'b', 'r','g', 'b', 'r','g', 'b', 'r']
#             ax: Axes
#             if not isinstance(option_event, AnalyseFromLabview):
#                 for idx, ax in enumerate(axis_list):
#                     ax.plot(np.array(norme_vecteur.loc[:, [items_norme_vecteur[idx]]]), colors[idx])
#                     # ax.set(xlim=(000, 700), ylim=(20, 80))
#                     ax.set(ylim=(0, 200))
#                     ax.set_xlabel(items_norme_vecteur[idx])
#             else:
#                 for idx, ax in enumerate(axis_list):
#                     ax.plot(np.array(norme_vecteur.loc[:, [items_norme_vecteur[idx]]]), colors[idx])
#                     ax = option_event.plot_event(idx, ax)
#                     # for i in option_event.reward_from_txt.index:
#                     #     ax.plot(
#                     #         [option_event.time_structure.loc[i, 'time_event_x'], option_event.time_structure.loc[i, 'time_event_x']],
#                     #         [option_event.time_structure.loc[i, 'time_event_y'], option_event.time_structure.loc[i, 'time_event_y'] - 100], colors[idx] + 'o-')
#                     #     # ax.set(xlim=(000, 700), ylim=(20, 80))
#                     ax.set(ylim=(norme_vecteur[items_norme_vecteur[idx]].min(), norme_vecteur[items_norme_vecteur[idx]].max()))
#                     ax.set_xlabel(items_norme_vecteur[idx])
#             fig1.show()
#         except TypeError:
#             raise TypeError('Ce n"est pas un DataFrame')
#         except ValueError:
#             raise ValueError('Problème de valeurs d"entrée')
#
#     def plot_acceleration(self, DeltaV: DataFrame):
#         fig1 = BasicPlot.crea_fig()
#         items_acceleration: List[str] = list(DeltaV.keys())
#         axis_list = BasicPlot.axis_list(items_acceleration, fig1)
#         colors =['g', 'b', 'r','g', 'b', 'r','g', 'b', 'r']
#         ax: Axes
#         for idx, ax in enumerate(axis_list):
#             ax.plot(np.array(DeltaV.loc[:, [items_acceleration[idx]]]), colors[idx])
#             ax.set(ylim=(-50, 50))
#             ax.set_xlabel(items_acceleration[idx])
#         fig1.show()
#
#     def plot_zone_brute(self, data: DataFrame, items: List[str]):
#         fig, ax = plt.subplots(1, figsize=(20, 20))
#         ax.set(xlim=(0, 800), ylim=(0, 800))
#         colors = ['g', 'b', 'r']
#
#         # ax.plot(data.loc[[0], [items[1]]], data.loc[[0], [items[2]]], 'bo', markersize=12)
#         # ax.plot(self.data.loc[[0], [self.items[4]]], self.data.loc[[0], [self.items[5]]], 'bo', markersize=12)
#         # ax.plot(self.data.loc[[0], [self.items[7]]], self.data.loc[[0], [self.items[8]]], 'ro', markersize=12)
#         # ax.plot(self.data.loc[[0], [self.items[10]]], self.data.loc[[0], [self.items[11]]], 'go', markersize=12)
#         # ax.plot(self.data.loc[[0], [self.items[13]]], self.data.loc[[0], [self.items[14]]], 'bo', markersize=12)
#         # ax.plot(self.data.loc[[0], [self.items[16]]], self.data.loc[[0], [self.items[17]]], 'bo', markersize=12)
#         # ax.plot(self.data.loc[[0], [self.items[19]]], self.data.loc[[0], [self.items[20]]], 'bo', markersize=12)
#         # ax.plot(self.data.loc[[0], [self.items[22]]], self.data.loc[[0], [self.items[23]]], 'bo', markersize=12)
#         for i in range(0,2000,20):
#             x = [np.array(data.loc[[i], [items[7]]])[0][0], np.array(data.loc[[i], [items[10]]])[0][0]]
#             y = [np.array(data.loc[[i], [items[8]]])[0][0], np.array(data.loc[[i], [items[11]]])[0][0]]
#
#             ax.plot(x,y)
#
#         fig.show()
#
#     def plot_vecteurs(self, data: DataFrame, items: List[str]):
#         fig, ax = plt.subplots(1, figsize=(20, 20))
#         ax.set(xlim=(0, 800), ylim=(0, 800))
#         colors = ['g', 'b', 'r']
#
#         # ax.plot(data.loc[[0], [items[1]]], data.loc[[0], [items[2]]], 'bo', markersize=12)
#
#         # for i in range(0,2000,20):
#         for i in range(0, 2, 1):
#             ax.plot(data.loc[[i], [items[7]]], data.loc[[i], [items[8]]], 'yo', markersize=12)
#
#             ABx = [np.array(data.loc[[i], [items[1]]])[0][0], np.array(data.loc[[i], [items[4]]])[0][0]]
#             ABy = [np.array(data.loc[[i], [items[2]]])[0][0], np.array(data.loc[[i], [items[5]]])[0][0]]
#             ax.plot(ABx,ABy,'b')
#             ACx = [np.array(data.loc[[i], [items[1]]])[0][0], np.array(data.loc[[i], [items[7]]])[0][0]]
#             ACy = [np.array(data.loc[[i], [items[2]]])[0][0], np.array(data.loc[[i], [items[8]]])[0][0]]
#             ax.plot(ACx,ACy,'r')
#             BCx = [np.array(data.loc[[i], [items[4]]])[0][0], np.array(data.loc[[i], [items[7]]])[0][0]]
#             BCy = [np.array(data.loc[[i], [items[5]]])[0][0], np.array(data.loc[[i], [items[8]]])[0][0]]
#             ax.plot(BCx,BCy,'g')
#
#         fig.show()
#
#     def plot_event_around(self, items_norme_vecteur: List[str], around_event: DataFrame):
#         """
#         les données doivent être préparées
#         :param items_norme_vecteur:
#         :param around_event:
#         :return:
#         """
#         fig1 = BasicPlot.crea_fig()
#         deb = list(range(0, len(around_event), 10))
#         fin = list(range(9, len(around_event), 10))
#         event_mean = [np.array(around_event.loc[d:f].mean()) for d, f in zip(deb, fin)]
#         event_std = [np.array(around_event.loc[d:f].std()) for d, f in zip(deb, fin)]
#         axis_list = BasicPlot.axis_list(items_norme_vecteur, fig1)
#         colors = ['g', 'b', 'r', 'g', 'b', 'r', 'g', 'b', 'r']
#         ax: Axes
#         for idx, ax in enumerate(axis_list):
#             y = np.array(event_mean[idx])
#             x = list(range(-20, 21, 1))
#             std = np.array(event_std[idx])
#             ax.errorbar(x, y, std, linestyle='-', marker='^')
#             ax.plot([0, 0],[0,20], colors[idx] + 'o-')
#             ax.set(ylim=(60, 300))
#             ax.set_xlabel(items_norme_vecteur[idx])
#         fig1.show()
#
#     def plot_aire(self, aire: DataFrame):
#         fig1 = BasicPlot.crea_fig()
#         items_triangle = list(aire.keys())
#         axis_list = BasicPlot.axis_list(items_triangle, fig1)
#         colors = ['g', 'b', 'r', 'g', 'b', 'r', 'g', 'b', 'r']
#         ax: Axes
#         for idx, ax in enumerate(axis_list):
#             ax.plot(aire[items_triangle], colors[idx])
#             ax.set(ylim=(0, 300))
#             ax.set_xlabel(items_triangle[idx])
#         fig1.show()
#
#     def plot_serie(self, serie: Series):
#         fig1 = BasicPlot.crea_fig()
#         items = ['series']
#         axis_list = BasicPlot.axis_list(items, fig1)
#         colors = ['g', 'b', 'r', 'g', 'b', 'r', 'g', 'b', 'r']
#         ax: Axes
#         for idx, ax in enumerate(axis_list):
#             ax.plot(serie.loc[:], colors[idx])
#             ax.set(ylim=(0, 200))
#             # ax.set_xlabel(items_triangle[idx])
#         fig1.show()


# class BasicTraitmentTrajectory(object):
#
#     def __init__(self):
#         self.traitment: BasicCorrectionTrajectory = BasicCorrectionTrajectory()
#
#     def correction(self, data_struct: AnalyseFromLabview, data_traj: DataFrame):
#         """
#
#         :param data_struct:
#         :param data_traj:
#         :return:
#         """
#         couple: List[str]
#         for i, couple in enumerate(data_struct.couple_de_points):
#             data_traj = self.traitment.correction_trajectoire(data_traj, couple)
#         return data_traj
#
#     def make_segment(self, data_traj: DataFrame, pointA: List, pointB: List, pointC: List) -> None:
#         # pointA = data_DLC.couple_de_points[0]
#         # pointB = data_DLC.couple_de_points[1]
#         # pointC = data_DLC.couple_de_points[2]
#         triangle_ABC = list(combi([pointA, pointB, pointC], 2))
#         for couple in triangle_ABC:
#             print(couple[0], couple[1])
#             self.traitment.norme_vecteur_entre_points(data=data_traj, couple_pointsA=couple[0], couple_pointsB=couple[1])
#
#
# class BasicCorrectionTrajectory(object):
#     """
#     cette class permet de calcul les normes des vecteurs.
#     si on rentre la trajectoire (labview) on aura la vitesse instannée.
#     la trajectoire est aussi corrigée. Dans le cas ou les couples de points sont le reflet de frame.
#     sinon cela représente la longueur des vecteurs
#     Pour utiliser cette class :
#
#     """
#     def __init__(self):
#         self.norme_vecteur: DataFrame = pd.DataFrame()
#         self.DeltaV: DataFrame = pd.DataFrame()
#         self.norme_vecteur_entre_point: DataFrame = pd.DataFrame()
#         self.coordonnee_vecteur_entre_point: DataFrame = pd.DataFrame()
#         self.around_event: DataFrame = pd.DataFrame()
#
#
#     def _calcul_norme_(self, x1: Series, x2: Series, y1: Series, y2: Series) -> Series:
#         """
#         calcul de la norme du vecteur ou de la longueur du vecteur
#         :param x1:
#         :param x2:
#         :param y1:
#         :param y2:
#         :return:
#         """
#         norme_vecteur: Series = pd.Series(np.sqrt((x2-x1)**2+(y2-y1)**2), name='norme_du_vecteur')
#         return norme_vecteur
#
#     def _coordonnee_vecteur_(self, x1: Series, x2: Series, y1: Series, y2: Series) -> Series:
#         """
#         calcul de la norme du vecteur ou de la longueur du vecteur AB
#         :param x1:
#         :param x2:
#         :param y1:
#         :param y2:
#         :return:
#         """
#         coordonnee_vecteur: Series = pd.Series([(x2 - x1), (y2 - y1)], name='coordonnee_vecteur', dtype=float)
#         return coordonnee_vecteur
#
#     def _norme_vecteur_(self, data: DataFrame, couple_pointsA: List[str], couple_pointsB: List[str] = None,
#                         index0: RangeIndex = None, index1: RangeIndex = None) -> Union[Series, Norme_Coordonnee_Vecteur]:
#         """
#         calcul de la longueur du vecteur entre deux frame pour le même point
#         donc de la norme du vecteur ||couple_pointsA[0]couple_pointsA[1]||
#         ou entre deux points dans la même frame
#
#         :param data:
#         :param items:
#         :return:
#         """
#         index_global = pd.RangeIndex(0, len(data.loc[:, [couple_pointsA[0]]])-1, 1)
#         if not isinstance(couple_pointsB, List):
#             if not isinstance(index0, RangeIndex) and not isinstance(index1, RangeIndex):
#                 print('Il manque une entrée !!!')
#             else:
#                 norme_AB = pd.DataFrame()
#                 idx = 0
#                 for _0, _1 in zip(couple_pointsA, couple_pointsA):
#                     idx +=1
#                     norme_AB[_0[-1]+str(idx)] = pd.Series(data[_0][index0])
#                     norme_AB[_1[-1]+str(idx+1)] = pd.Series(np.array(data[_1][index1]))
#                     idx = 0
#                 items = norme_AB.keys()
#
#                 norme_vecteur = self._calcul_norme_(norme_AB[items[0]], norme_AB[items[1]], norme_AB[items[2]],
#                                                        norme_AB[items[3]])
#                 self.norme_vecteur[couple_pointsA[0][:-2]] = norme_vecteur
#                 return norme_vecteur
#         else:
#             c = []
#             [c.extend([a, b]) for a, b in zip(couple_pointsA, couple_pointsB)]
#             norme_AB = pd.DataFrame(columns=c)
#
#             for x, y in zip(couple_pointsA, couple_pointsB):
#                 norme_AB[x] = pd.Series(data[x], data.index)
#                 norme_AB[y] = pd.Series(data[y], data.index)
#             items = norme_AB.keys()
#
#             coordonnee_vecteur_entre_point = self._coordonnee_vecteur_(norme_AB[items[0]], norme_AB[items[1]], norme_AB[items[2]],
#                                                    norme_AB[items[3]])
#             norme_vecteur_entre_point = self._calcul_norme_(norme_AB[items[0]], norme_AB[items[1]], norme_AB[items[2]],
#                                                    norme_AB[items[3]])
#
#             self.norme_vecteur_entre_point[couple_pointsA[0][0:5] + ' ' + couple_pointsB[0][0:5]] = norme_vecteur_entre_point
#             self.coordonnee_vecteur_entre_point[couple_pointsA[0][0:5] + ' ' + couple_pointsB[0][0:5]] = coordonnee_vecteur_entre_point
#
#             return norme_vecteur_entre_point, coordonnee_vecteur_entre_point
#
#     def _seuillage_norme_vecteur_(self, data: DataFrame, seuil: int, norme_vecteur: Series, items: list):
#         """
#         Si des points aberrants sont détecter on lance la correction et on enregistre
#         :param data:
#         :param seuil:
#         :param norme_vecteur:
#         :param items:
#         :return:
#         """
#         point_aberrant = norme_vecteur[norme_vecteur > seuil]
#         if len(point_aberrant) > 0:
#             start, stop = self._extract_start_stop_(point_aberrant)
#
#             _cahe_data_work_ = self._interpolation_val_aberrant_(data, start, stop, items[0], items[1])
#             return _cahe_data_work_
#         else:
#             return 0
#
#     def _extract_start_stop_(self, point_aberrant: Series) -> Tuple[List[int], List[int]]:
#             n: int = len(point_aberrant)
#             init: int = 0
#             start: List[int] = []
#             stop: List[int] = []
#             self._start_stop_recur_(n, init, point_aberrant, start, stop)
#             return start, stop
#
#     def _start_stop_recur_(self, n: int, init: int, point_aberrant: Series, start: list, stop: list):
#         """
#         fonction recurante qui le début et la fin de chaque séquence de point abérrant
#         :param n:
#         :param init:
#         :param point_aberrant:
#         :param start:
#         :param stop:
#         :return:
#         """
#         if n == 0 or init >= len(point_aberrant)-1:
#             print(n)
#             return start, stop
#         else:
#             idx = 0
#             on = []
#             while point_aberrant.index[init] + idx in point_aberrant.index:
#                 idx += 1
#                 on.append(point_aberrant.index[init]+idx)
#             if len(start) < 1:
#                 start.append(point_aberrant.index[init] - 1)
#                 stop.append(point_aberrant.index[init] + idx + 1)
#                 init = init + idx
#             else:
#                 if point_aberrant.index[init] - 1 <= stop[-1]:
#                     stop[-1] = point_aberrant.index[init] + idx + 1
#                     init = init + idx
#                 else:
#                     start.append(point_aberrant.index[init] - 1)
#                     stop.append(point_aberrant.index[init] + idx + 1)
#                     init = init + idx
#             return self._start_stop_recur_(n - 1, init, point_aberrant, start, stop)
#
#     def _interpol_val_recur_(self, data: DataFrame, n: int, x: list, y: list, init: int, start: list, stop: list, item_x: str, item_y: str) -> tuple:
#         if n == 0:
#             return x, y
#         else:
#             _x = np.array(data.loc[(range(start[init], stop[init])), [item_x]])
#             _x = np.array([i[0] for i in _x])
#             longueur_x = len(_x)
#             _y = np.array(data.loc[(range(start[init], stop[init])), [item_y]])
#             _y = np.array([i[0] for i in _y])
#
#             f = interp1d(_x, _y)
#             _x_interm = np.linspace(_x[0], _x[-1], num=longueur_x, endpoint=True)
#             _y_interm = f(_x_interm)
#             x.extend(_x_interm.tolist())
#             y.extend(_y_interm.tolist())
#             init += 1
#             return self._interpol_val_recur_(data, n-1, x, y, init, start, stop, item_x, item_y)
#
#     def _interpolation_val_aberrant_(self, data: DataFrame, start: list, stop: list, item_x: str, item_y: str) -> DataFrame:
#         """
#         on remplace les valeurs corrigées directement dans la series correspondant aux points (x,y) du couple entrée (traj).
#
#         :param data:
#         :param start:
#         :param stop:
#         :param item_x:
#         :param item_y:
#         :return:
#         """
#         n = len(start)
#         x: list = []
#         y: list = []
#         init: int = 0
#         self._interpol_val_recur_(data, n, x, y, init, start, stop, item_x, item_y)
#         index = []
#         [index.extend(list(range(start[i], stop[i]))) for i in range(len(start))]
#         yvals = pd.Series(y, index, float)
#         xvals = pd.Series(x, index, float)
#         _cahe_work_data_ = data
#         _cahe_work_data_.loc[xvals.index, [item_x]] = xvals
#         _cahe_work_data_.loc[yvals.index, [item_y]] = yvals
#         return _cahe_work_data_
#
#     def _calcul_acceleration_(self, norme_vecteur: DataFrame, items_norme_vecteur: str) -> Series:
#         """
#         instatanée
#         vitesse = norme_vecteur
#         vitesse = V
#         temps = T
#         acceleration = delta V / delta T. delta V = V final - V initial, delta T = T final - T initial
#         delta T = 1 donc acceleration = V final - V initial
#
#         :return:
#         """
#         t = list(range(1, len(norme_vecteur.loc[:, [items_norme_vecteur]]), 1))
#         e = list(range(0, len(norme_vecteur.loc[:, [items_norme_vecteur]]) - 1, 1))
#
#         _Vinitial = np.array([i[0] for i in np.array(norme_vecteur.loc[e, [items_norme_vecteur]])])
#         Vinitial = pd.Series(_Vinitial, list(range(0, len(norme_vecteur.loc[:, [items_norme_vecteur]])-1)))
#         _Vfinal = np.array([i[0] for i in np.array(norme_vecteur.loc[t, [items_norme_vecteur]])])
#         Vfinal = pd.Series(_Vfinal, list(range(0, len(norme_vecteur.loc[:, [items_norme_vecteur]])-1)))
#
#         DeltaV: Series = Vfinal-Vinitial
#         return DeltaV
#
#     def acceleration_instant(self):
#         items_norme_vecteur: List[str] = list(self.norme_vecteur.keys())
#         for i in range(len(items_norme_vecteur)):
#             DeltaV = self._calcul_acceleration_(self.norme_vecteur, items_norme_vecteur[i])
#             self.DeltaV[items_norme_vecteur[i]] = DeltaV
#
#     def correction_trajectoire(self, data: DataFrame, couple_pointsA: List[str]):
#         """
#
#         :param data:
#         :param couple_pointsA: correspond à un couple de point de coordonnées ['earr x', 'earr y']
#         :return:
#         """
#         index1 = pd.RangeIndex(1, len(data.loc[:, [couple_pointsA[0]]]), 1)
#         index0 = pd.RangeIndex(0, len(data.loc[:, [couple_pointsA[0]]]) - 1, 1)
#         # step 1
#         norme_vecteur = self._norme_vecteur_(data=data, couple_pointsA=couple_pointsA, index0=index0, index1=index1)
#         seuil: int = 40
#         # step 2
#         new_data = data.copy()
#         _cache_data_work_ = self._seuillage_norme_vecteur_(data, seuil, norme_vecteur, couple_pointsA)
#         if isinstance(_cache_data_work_, DataFrame):
#             new_data = _cache_data_work_
#         # step 3
#         norme_vecteur = self._norme_vecteur_(data=new_data, couple_pointsA=couple_pointsA, index0=index0, index1=index1)
#         print('correction ok')
#         return new_data
#         #TODO a réécrire en  numpy
#
#     def norme_vecteur_entre_points(self, data: DataFrame, couple_pointsA: List[str], couple_pointsB: List[str]) -> Norme_Coordonnee_Vecteur:
#         """
#         calcul de la longueur du vecteur entre deux points  donc de la norme du vecteur ||itemsAitemsB|| pour chaque frame
#
#         :param couple_pointsB:
#         :param couple_pointsA:
#         :param data:
#         :return:
#         """
#
#         norme_vecteur_entre_point = self._norme_vecteur_(data=data, couple_pointsA=couple_pointsA, couple_pointsB=couple_pointsB)
#         return norme_vecteur_entre_point


# class AnimalSurface(object):
#     def __init__(self):
#         self.AB: Series = None
#         self.AC: Series = None
#         self.BC: Series = None
#         self.AH: tuple = None
#         self.aire: DataFrame = pd.DataFrame()
#         self.ABAC: Series = None
#         self.angle_ACB: Series = None
#         self.GAMMA_angle_BAC2: Series = None
#         self.beta_angle_ABC2: Series = None
#         self.alpha_angle_ACB2: Series = None
#
#         self.ABBC: Series = None
#         self.ACBC: Series = None
#     def calcul_de_laire_by_heron(self, AC: Series, AB: Series, BC: Series):
#         """
#         A**2 = s*(s-AB)*(s-BC)*(s-CA)
#         donc : s = (1/2)*périmétre = (1/2)*(AB+BC+AC)
#
#         :param AC:
#         :param AB:
#         :param BC:
#         :return:
#         """
#         s = (1/2)*(AB+BC+AC)
#         _A = s * (s - AB) * (s - BC) * (s - AC)
#         A = np.sqrt(_A)
#         return A
#
#     def calcul_de_aire(self, data: BasicCorrectionTrajectory, items_triangle: str):
#         """
#         AB.AC = AB.AH car H est la projection orthogonale de C
#         :return:
#         """
#         items_norme = list(data.norme_vecteur_entre_point.keys())
#         AC = data.norme_vecteur_entre_point[items_norme[1]]
#         AB = data.norme_vecteur_entre_point[items_norme[0]]
#         BC = data.norme_vecteur_entre_point[items_norme[2]]
#
#         # self._calcul_des_normes_(AC, AB, BC)
#         self.aire[items_triangle] = self.calcul_de_laire_by_heron(AC=AC, AB=AB, BC=BC)
#
#         # angle_BAC = self._m2_calcul_angle_ACB_(BC=BC, AC=AC, AB=AB)
#         # ABAC = self._calcul_ps_by_length(AC=AC, AB=AB, BC=BC)
#         # AH = self._calcul_AH_by_projet_orthogonal(ABAC, AB, angle_BAC)
#         # self.aire[items_triangle] = self._aire_triangle_(AB, AH)
#         print('fin')
#
#     def _calcul_ps_by_length(self, AC: Series, AB: Series, BC: Series):
#         """
#         AB.AC = (1/2)*(AB**2+AC**2-BC**2)
#
#         :return:
#         """
#         self.ABAC = (1/2)*(AB**2+AC**2-BC**2)
#         self.ABBC = (1/2)*(AB**2+BC**2-AC**2)
#         self.ACBC = (1/2)*(BC**2+AC**2-AB**2)
#
#         return self.ABAC
#
#     def _calcul_ps_by_coordonnee(self, data):
#         """
#         AB(x,y);AC(x',y')
#         :param data:
#         :return:
#         """
#         items_coordonnee = list(data.coordonnee_vecteur_entre_point.keys())
#         x = pd.Series(data.coordonnee_vecteur_entre_point[items_coordonnee[0]][0])
#         x_p = pd.Series(data.coordonnee_vecteur_entre_point[items_coordonnee[0]][1])
#         y = pd.Series(data.coordonnee_vecteur_entre_point[items_coordonnee[1]][0])
#         y_p = pd.Series(data.coordonnee_vecteur_entre_point[items_coordonnee[1]][1])
#         self._calcul_produit_scalaire_(x, x_p, y, y_p)
#
#     def _calcul_AH_by_projet_orthogonal(self, ABAC, AB, angle_bac):
#         """
#         AB.AC = AB*AH = ||AB||*||AH|| car AB.AC = AB*AC PAR LA PROJECTION ORTHOGONAL
#         donc
#         AH = AB.AC/AB colineaire et de même direction et sens
#         AH = AB.AC/-AB colineaire et de même direction et sens
#
#         :param ABAC:
#         :param AB:
#         :return:
#         """
#         _AH = np.array([])
#
#         for i, val in enumerate(angle_bac):
#             if angle_bac[i] < 45:
#                 _tmp = -ABAC[i]/AB[i]
#             else:
#                 _tmp = ABAC[i]/AB[i]
#             _AH = np.hstack((_AH, _tmp))
#         AH = pd.Series(_AH)
#         return AH
#
#     # def _calcul_des_normes_(self, AC: Series, AB: Series, BC: Series) -> bool:
#     #     """
#     #     calcul la longueur ou la norme de chaque vecteur (ou côté du triangle)
#     #     AC,AB,BC sont de type tuple avec en première series les normes et en deuxième les coordonnées vectorieles
#     #     :param data:
#     #     :return:
#     #     """
#     #     self.AC = AC
#     #     self.AB = AB
#     #     self.BC = BC
#     #     print('ok')
#     #     return True
#
#     # def _calcul_ps_ABAC_(self, x, x_p, y, y_p):
#     #     """
#     #     calcul du produit scalaire AB.AC
#     #     AB.AC = xx' +yy'
#     #
#     #     :param x:
#     #     :param x_p:
#     #     :param y:
#     #     :param y_p:
#     #     :return:
#     #     """
#     #     abac = x*x_p + y*y_p
#     #     print('Produit scalaire de AB.AC')
#     #     return abac
#
#     def _calcul_produit_scalaire_(self, x: Series, xprime: Series, y: Series, yprime: Series) -> Series:
#         """
#         calcul du produit scalaire AB.AC
#         AB.AC = xx' +yy'
#
#         :param x:
#         :param x_p:
#         :param y:
#         :param y_p:
#         :return:
#         """
#         produit_scalaire = x*xprime + y*yprime
#         print('Produit scalaire de AB.AC')
#         return produit_scalaire
#
#     # def _norme_AH_(self, AB: Series, ABAC: Series, angle_ACB) -> Series:
#     #     _AH = np.array([])
#     #     for idx, val in enumerate(angle_ACB):
#     #         if val < 90:
#     #             _tmp = ABAC[idx]/-AB[idx]
#     #         else:
#     #             _tmp = ABAC[idx]/AB[idx]
#     #         _AH = np.hstack((_AH, _tmp))
#     #     AH = pd.Series(_AH)
#     #     print(('norme AH'))
#     #     return AH
#
#     def _colinearite_(self):
#         """
#         Soient les vecteurs de coordonnées u(x; y) et v(x'; y').
#         Les vecteurs u et v sont colinéaire si et seulement si : xy' - yx' = 0
#
#         """
#         pass
#
#     def _aire_triangle_(self, AB: Series, AH: Series) -> Series:
#         aire = (AB*AH)/2
#         print('aire')
#         return aire
#
#     # def _calcul_angle_ACB_(self, BC: Series, AC: Series, AB: Series):
#     #     """
#     #     theoreme de al-kashi
#     #     a**2 = b**2+c**2-2bc.cos(alpha)
#     #     etcs
#     #     a = CB
#     #     c = AB
#     #     b = AC
#     #     angle bc (
#     #     alpha = arcos[(b**2+c**2-a**2)/2bc]
#     #     angle ca
#     #     béta = arcos[(a**2+c**2-b**2)/2ac]
#     #     angle ab
#     #     gamma = arcos[(a**2+b**2-c**2)/2ab]
#     #
#     #     :return:
#     #     """
#     #     _angle_ACB = np.arccos(np.array((AC**2 +AB**2 - BC**2) / (2*AB*AC)))
#     #     angle_ACB = pd.Series(np.array((_angle_ACB*180)/(np.pi)), name='angle ACB')
#     #     print(angle_ACB.max())
#     #     return angle_ACB
#
#     def _m2_calcul_angle_ACB_(self, BC: Series, AC: Series, AB: Series):
#         """
#         theoreme de al-kashi
#         a**2 = b**2+c**2-2bc.cos(alpha)
#         etcs
#         a = CB
#         c = AB
#         b = AC
#         angle bc (
#         alpha = arcos[(b**2+c**2-a**2)/2bc]
#         angle ca
#         béta = arcos[(a**2+c**2-b**2)/2ac]
#         angle ab
#         gamma = arcos[(a**2+b**2-c**2)/2ab]
#
#         :return:
#         """
#         _GAMMA_angle_ACB2 = np.arccos(np.array((BC ** 2 + AC ** 2 - AB ** 2) / (2 * BC * AC)))
#         self.GAMMA_angle_ACB2 = pd.Series(np.rad2deg(_GAMMA_angle_ACB2), name='angle BAC')
#
#         _beta_angle_ABC2 = np.arccos(np.array((BC ** 2 + AB ** 2 - AC ** 2) / (2 * BC * AB)))
#         self.beta_angle_ABC2 = pd.Series(np.rad2deg(_beta_angle_ABC2), name='angle ABC')
#
#         _alpha_angle_BAC2 = np.arccos(np.array((AC**2 +AB**2 - BC**2) / (2*AB*AC)))
#         self.alpha_angle_BAC2 = pd.Series(np.rad2deg(_alpha_angle_BAC2), name='angle ACB')
#
#         # _angle_ACB = np.cos(self.ABAC/(AB*AC))
#         # _angle_ACB = np.cos(self.ABAC/(AB*AC))
#         # _angle_ACB = np.cos(self.ABAC/(AB*AC))
#
#
#         # # _angle_ACB = np.arccos(np.array((BC**2 +AC**2 - AB**2) / (2*BC*AC)))
#         # angle_ACB = pd.Series(np.array((_angle_ACB*180)/(np.pi)), name='angle ACB')
#         # print(angle_ACB.max())
#         return self.GAMMA_angle_BAC2


if __name__ == '__main__':
    """
    
    dir_txt_traj = r'D:\Dropbox\python\import_neuralynxv2\data\cplx07 + bsl\fichier traj\*.txt'

    # chargement du fichier trajectoire contenant "x, y, point"
    trajectoire = LoadData.init_data(LabviewFilesTrajectory, dir_txt_traj)
    
    # chargement des temps en ms "temps" contenu dans le fichier rewards
    reward = LoadData.init_data(LabviewFilesReward, dir_txt_traj)

    - Analyse FromLabview permet de d'extraire les rewards et les omissions caller sur la base_time 
    de la vidéo.
    ps: les temps des "reward" sont recallé sur les temps de passage des points, 
    si on veut les temps originaux des rewards, il faut prendre "rewards_from_txt"
        - l'ensemble des données est accéssible par all_data contenu dans data_AFL
    data_AFL = AnalyseFromLabview()
    # création d'un dataframe contenant "trajectoire_x, trajectoire_y, reward, rewards, omissions"
    data_tracking_AFL = data_AFL.load_data_from_labview(trajectoire, reward)
    """

    s1 = chrono.time()
    print('start')

    # dir_txt = r'/data/cplx07 + bsl/fichier traj/shenron1_cplx_07_filter1_08042020-1527_reward.txt'
    # dir_txt_traj = r'D:\Dropbox\python\import_neuralynxv2\data\cplx07 + bsl\fichier traj\shenron1_cplx_07_filter1_08042020-1527traj.txt'

    dir_txt_traj = r'D:\Dropbox\python\import_neuralynxv2\data\cplx07 + bsl\fichier traj\*.txt'

    # chargement de la trajectoire contient "x, y, point"
    trajectoire = LoadData.init_data(LabviewFilesTrajectory, dir_txt_traj)
    # chargement des temps en ms "temps" contenu dans le fichier rewards
    reward = LoadData.init_data(LabviewFilesReward, dir_txt_traj)

    # traitement
    data_AFL = AnalyseFromLabview()
    # création d'un dataframe contenant "trajectoire_x, trajectoire_y, reward, rewards, omissions"
    data_tracking_AFL = data_AFL.load_data_from_labview(trajectoire, reward)


    #-----------------------------------------------------------------------------------
    # x = pd.Series([2, 5, 4], name='x')
    # y = pd.Series([3, 1, 4], name='y')

    # data_tracking_test = pd.DataFrame({'Ax': [2,2,2,2], 'Ay': [1,1,1,1], 'Bx': [5,5,5,5], 'By': [1,1,1,1], 'Cx': [0.5,6,5,2], 'Cy': [5,5,5,5]}, index=[0,1,2,3])
    # items_test = list(data_tracking_test)
    #
    # data_test = AnalyseFromLabview()
    # data_test.couple_de_points = [[items_test[0], items_test[1]], [items_test[2], items_test[3]], [items_test[4], items_test[5]]]
    #
    #
    # traitment_test = BasicTraitmentTrajectory()
    # data_tracking_test = traitment_test.correction(data_test, data_tracking_test)
    # #
    # traitment_test.make_segment(data_traj=data_tracking_test, pointA=data_test.couple_de_points[0],
    #                             pointB=data_test.couple_de_points[1], pointC=data_test.couple_de_points[2])



    # traitment_test2 = BasicTraitmentTrajectory()
    # traitment_test2.traitment.norme_vecteur_entre_point = pd.DataFrame({traitment_test.traitment.norme_vecteur_entre_point.columns[0]:8,
    #                                                                     traitment_test.traitment.norme_vecteur_entre_point.columns[1]:4,
    #                                                                     traitment_test.traitment.norme_vecteur_entre_point.columns[2]:6},index=[0])
    # traitment_test2.traitment.coordonnee_vecteur_entre_point = pd.DataFrame({traitment_test.traitment.norme_vecteur_entre_point.columns[0]:1,
    #                                                                     traitment_test.traitment.norme_vecteur_entre_point.columns[1]:1,
    #                                                                     traitment_test.traitment.norme_vecteur_entre_point.columns[2]:1},index=[0])


    # items_triangle = 'tete'
    # surface = AnimalSurface()
    # surface.calcul_de_aire(traitment_DLC.traitment, items_triangle)
    # traitment_DLC.traitment.make_event_around(surface.aire, data_AFL)


    # # data.acceleration_instant()
    # # data.make_event_around(data.norme_vecteur, reward)
    # # #
    # plotcomportement = SpecifiquePlot()
    # # # plotcomportement.plot_serie(serie=surface.angle_ACB)
    # plotcomportement.plot_aire(surface.aire)
    # plotcomportement.plot_vecteurs(data_tracking_test, list(data_tracking_test.columns))
    # plotcomportement.plot_event_around(['tete'], traitment_DLC.traitment.around_event)
    # plotcomportement.plot_norme_vecteur(surface.aire, data_AFL)
    # # plotcomportement.plot_acceleration(data.DeltaV)
    # # plotcomportement.plot_norme_vecteur(data.norme_vecteur_entre_point, reward)
    # # plotcomportement.plot_zone_brute()

    # # plotcomportement.plot_traj(data_tracking_DLC,
    # #                            [data_DLC.items[1], data_DLC.items[2]],
    # #                            [data_DLC.items[4], data_DLC.items[5]],
    # #                            [data_DLC.items[7], data_DLC.items[8]])
    # plotcomportement.plot_norme_vecteur(traitment_DLC.traitment.norme_vecteur, data_AFL)
    #
    # plotcomportement.plot_traj(data_tracking_AFL,
    #                            data_AFL.couple_de_points[0])
    # plotcomportement.plot_norme_vecteur(traitment_AFL.traitment.norme_vecteur, data_AFL)
    print('fin')
    s2 = chrono.time()
    print(f'temps écoulé : {s2 - s1}')