# -*-coding:utf-8 -*-
# traitement_deeplabcut package --> DeepLabCut_traitement module

import pandas as pd
from pandas import DataFrame, Series, RangeIndex, Index
from itertools import combinations as combi
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
from py_NPClab_Package.utilitaire_load.basic_load import LabviewFilesReward, LabviewFilesTrajectory, LoadData, DeepLabCutFileImport
from py_NPClab_Package.utlilitaire_saving.Saving_traitment import SaveSerialisation

import logging
from collections import defaultdict, OrderedDict, ChainMap


Norme_Coordonnee_Vecteur = Tuple[Series, Series]
Trajectoire = Tuple[ndarray, ndarray, ndarray]

from abc import ABC, abstractmethod, abstractclassmethod


class PrePlot(ABC):

    # @abstractmethod
    # def _couple_items_(self, items: List[str]) -> List[List[str]]:
    #     pass

    @abstractmethod
    def _make_good_format_data_(self, data_brute: DataFrame, option=None) -> DataFrame:
        pass


class ImportFiles(object):


    @staticmethod
    def load_txt_reward(dir: str) -> Series:
        reward_raw: ndarray = np.array([], dtype=int)
        print(f'path : {dir}')
        with open(dir, 'r') as file:
            for line in file:
                reward_raw = np.hstack((reward_raw, int(line.split('\t')[0])))
        event_time: Series = pd.Series(reward_raw)
        return event_time


    @staticmethod
    def load_txt_trajectoire_to_labview(dir: str) -> Trajectoire:
        trajectoire_x: ndarray = np.array([], dtype=float)
        trajectoire_y: ndarray = np.array([], dtype=float)
        reward_raw: ndarray = np.array([], dtype=int)

        print(f'path : {dir}')
        with open(dir, 'r') as file:
            for line in file:
                trajectoire_x = np.hstack((trajectoire_x, float(line.split('\t')[0].replace(',','.'))))
                trajectoire_y = np.hstack((trajectoire_y, float(line.split('\t')[1].replace(',','.'))))
                reward_raw = np.hstack((reward_raw, int(line.split('\t')[2].split(',')[0])))
        return trajectoire_x, trajectoire_y, reward_raw


class PreFormDLC(PrePlot):
    """
    L'architecture des fichiers DLC est :
        - pour chaque point demandé il y a 3 colonnes
        - chaque groupe de 3 correspond à " x, y, likehood"
        _ la première ligne peut être supprimé
        _ la deuxième "bodyparts" correspond au nom de chaque point
        - la troisième "coords" correspond au nom de la colonne "x" etcs
    """

    def __init__(self):
        self.data_brute: DataFrame = None
        self.data: defaultdict = defaultdict(DataFrame)
        self.items: List[str] = None
        # self.couple_de_points: List[List[str]] = None


    # def _couple_items_(self, items: List[str]) -> List[List[str]]:
    #     """
    #     le 25 correspond au nombre de colonne totale du dataframe brute
    #     :param items:
    #     :return:
    #     """
    #     idx = 0
    #     num_items = []
    #     couple_items = []
    #     for i in range(1, 25, 1):
    #         if idx == 2:
    #             idx = 0
    #         else:
    #             if len(num_items) == 2:
    #                 couple_items.append([items[num_items[0]], items[num_items[1]]])
    #                 num_items = []
    #                 num_items.append(i)
    #                 idx += 1
    #             else:
    #                 num_items.append(i)
    #                 idx += 1
    #     couple_items.append([items[num_items[0]], items[num_items[1]]])
    #     return couple_items

    def _bodyparts_(self, item_bodyparts, data_reformat: DataFrame, data_classified):
        """
        boucle recursive
        ps : même temps qu'une boucle "for"

        for indice in range(len(item_bodyparts)):
            data_classified[item_bodyparts[indice]] = data_reformat[item_bodyparts[indice]]
        return data_classified

        :param item_bodyparts:
        :return: defaultdict
        """
        # boucle récursive
        indice = 0
        if indice == len(item_bodyparts):
            return data_classified
        else:
            data_classified[item_bodyparts[indice]] = data_reformat[item_bodyparts[indice]]
            data_classified[item_bodyparts[indice]].columns = Index([i for i in data_reformat[item_bodyparts[indice]].iloc[0]])
            data_classified[item_bodyparts[indice]] = data_classified[item_bodyparts[indice]].drop([0], axis='rows')
            # data_classified[item_bodyparts[indice]] = data_classified[item_bodyparts[indice]].reset_index(drop=True, inplace=True)

            data_classified[item_bodyparts[indice]].reset_index(drop=True, inplace=True)
            data_classified[item_bodyparts[indice]] = data_classified[item_bodyparts[indice]].astype(dtype=float)
            indice += 1
            item_bodyparts = item_bodyparts[indice:]
            return self._bodyparts_(item_bodyparts, data_reformat, data_classified)



    def _make_good_format_data_(self, data_brute: DataFrame, option=None) -> defaultdict:
        """
        data_brute est un dataframe de l'ensemble du csv donc fonctionne comme un dict
        reconstruction du dataframe utilisable
        :return:
        """
        item_columns_init = [i for i in data_brute.keys()]

        item_bodyparts = [data_brute[item_columns_init[i]][0] for i in range(len(item_columns_init)) if data_brute[item_columns_init[i]][0].find('bodyparts') == -1]
        data_reformat = data_brute.drop(['scorer'], axis='columns')
        data_reformat = data_reformat.drop([0], axis='rows')

        data_reformat.reset_index(drop=True, inplace=True)
        data_reformat.columns = item_bodyparts
        data_classified = defaultdict(DataFrame)
        item_bodyparts = list(set(item_bodyparts))
        s3 = chrono.perf_counter()
        data_classified = self._bodyparts_(item_bodyparts,data_reformat, data_classified)
        s4 = chrono.perf_counter()
        logging.debug(f'durée classified: {s4 - s3}')
        self.items = item_bodyparts
        return data_classified

    def load_data_from_DLC(self, data_brute: DataFrame) -> defaultdict:
        self.data = self._make_good_format_data_(data_brute)
        return self.data

if __name__ == "__main__":
    dir_DLC = r'/data/trajectoire_tarek_deeplapcut/DLC_data'
    DLC_brute = LoadData.init_data(DeepLabCutFileImport, dir_DLC, 'Chichi_3_cplx_04bis_500mV_03042020-1627DLC_resnet50_testgpuMay2shuffle1_50000.csv')

    DLC = PreFormDLC()
    data = DLC.load_data_from_DLC(DLC_brute)