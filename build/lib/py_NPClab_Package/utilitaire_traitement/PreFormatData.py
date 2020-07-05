# -*-coding:utf-8 -*-

import pandas as pd
from pandas import DataFrame, Series, RangeIndex
from itertools import combinations as combi
import itertools as iter
import matplotlib.pyplot as plt
import time as chrono
import numpy as np
from numpy.core.multiarray import ndarray
from typing import List, Union, Tuple
import logging

Norme_Coordonnee_Vecteur = Tuple[Series, Series]
Trajectoire = Tuple[ndarray, ndarray, ndarray]


class PreFormatData(object):

    def make_event_around(self, data: DataFrame, reward: Series):
        """
        Pour utiliser cette methode il faut que la series des event soit indexé sur la base de temps
        des data pour en extraire les valeurs correctement
        calcul pour toutes les colonnes d'un dataframe
        :param data:
        :param reward: c'est les temps des event en ms sur lesquel on ce calle
        :return:
        """
        s3 = chrono.perf_counter()
        items = list(data.keys())
        n = len(data.columns)
        init = 0
        e = pd.DataFrame(dtype=float)
        e = self._around_event_recur_(n, data, e, reward, items, init)
        # event_mean_std = np.array([np.array(e.loc[:].mean()), np.array(e.loc[:].std())])
        self.around_event = e
        s4 = chrono.perf_counter()
        logging.debug(f'Fin Event, durée all via map: {s4 - s3}')
        return self.around_event

    def _around_event_recur_(self, n: int, data: DataFrame, e: DataFrame, reward: Series, items: List[str],
                             init: int):
        """
        extraction des valeurs autour des events pour une series
        :param n:
        :param data:
        :param e:
        :param reward_from_txt:
        :param items:
        :param init:
        :return:
        """

        if n == 0 and init == len(data.columns):
            return e
        else:
            for idx, num_reward_index in enumerate(reward.index):
                e = e.append(pd.Series(np.array(data.loc[num_reward_index - 20:num_reward_index + 20,
                    items[init]]), name=str(idx)), ignore_index=True)
            init += 1
            return self._around_event_recur_(n - 1, data, e, reward, items, init)

if __name__ == '__main__':
    pass