import logging
from collections import defaultdict
import numpy as np
from numpy.core.multiarray import ndarray
from pathlib import Path
from typing import List, Dict, Union, Tuple, Any
from py_NPClab_Package.utilitaire_neuralynx.Load_neuralynx import GlobalEventBasetime, GlobalEventSignal
from py_NPClab_Package.utilitaire_load.basic_load import LoadData, ImportNeuralynx
from py_NPClab_Package.utlilitaire_saving.Saving_traitment import SaveSerialisation
from py_NPClab_Package.utilitaire_traitement.ExtractionVal import ExtractEvent

import time as chrono

import pandas as pd
from pandas import DataFrame, Series
import re

Event_type = Dict[str, Union[GlobalEventSignal, GlobalEventBasetime]]

logging.basicConfig(level=logging.DEBUG)


class EventFromOther(ExtractEvent):
    """
    Cette class permet d'obtenir les valeurs d'event provenant d'une autre origine que neuralynx
    comme les
    """

    def __init__(self):
        super().__init__()
        # self.event_time: ndarray = event_time
        # self.base_time: Quantity = base_time
        self.event_index_in_raw: ndarray([int]) = np.array([], dtype=int)
        self.event_recaller_index_in_raw: ndarray([int]) = np.array([], dtype=int)


    def _pre_format_(self, time_event_ref_rmz: float, event_time: Union[ndarray, Series]) -> Series:
        """
        on utilise le "time_event_ref_rmz" pour recaller les events sur le même point de
        départ que la base de temps
        """
        if isinstance(event_time, Series):
            event_time = (event_time/1000) + time_event_ref_rmz
        else:
            event_time = pd.Series((event_time/1000) + time_event_ref_rmz, index=pd.RangeIndex(0, len(event_time)), dtype=float)
        return event_time

    def _rmz_base_time_(self, time_event_ref_rmz: float, base_time: ndarray) -> Series:
        """
        Pour l'extraction "base_time" doit être une series

        :param time_event_ref_rmz:
        :param base_time:
        :return:
        """
        # base_time = base_time - time_event_ref_rmz

        base_time = base_time
        base_time = pd.Series(base_time, dtype=float)
        return base_time

    def start(self, event_time: ndarray, base_time: ndarray, time_event_ref_rmz: float):
        time = self._pre_format_(time_event_ref_rmz, event_time)
        base_time = self._rmz_base_time_(time_event_ref_rmz, base_time)
        self.event_index_in_raw = self._extraction_event_times_(time, base_time)


class ClassificationPattern(SaveSerialisation):

    def _set_pattern_(self, name: str, time_up: float = None, dir_profile_pattern: str = None,
                      time_down: float = None, pattern: str = None, other_path: str = None) -> Union[Dict]:
        """
        le "A" correspond au time_up et le "B" au time_down
        :param name: correspond au nom du fichier (nom du profile)
        """
        if isinstance(other_path, type(None)) and not isinstance(dir_profile_pattern, type(None)):
            # dir_profile_pattern = other_path

            if isinstance(time_up, type(None)) and \
                isinstance(time_down, type(None)) and isinstance(pattern, type(None)):
                profile_pattern = self.load_data_serializer(path=dir_profile_pattern, name=name)
                logging.debug(f'Fichier chargé : {name}')
                return profile_pattern

            else:
                saving = self._get_conf_(dir_save_conf=dir_profile_pattern, name=name, name_folder='profile_pattern')
                if saving[0] == 1:
                    profile_pattern = {'time_up': time_up, 'time_down': time_down, 'pattern': pattern, 'name': name}
                    self._set_conf_(name=name, dir_save_conf=saving[1], data=profile_pattern)
                    logging.debug(f'Fichier sauvegardé : {name}')

        if not isinstance(other_path, type(None)) and isinstance(dir_profile_pattern, type(None)):
            dir_profile_pattern = other_path

            if isinstance(time_up, type(None)) and \
                    isinstance(time_down, type(None)) and isinstance(pattern, type(None)):
                profile_pattern = self.load_data_serializer(path=dir_profile_pattern, name=name)
                logging.debug(f'Fichier chargé : {name}')
                return profile_pattern

            else:
                saving = self._get_conf_(other_path=other_path, name=name, name_folder='profile_pattern')
                if saving[0] == 1:
                    profile_pattern = {'time_up': time_up, 'time_down': time_down, 'pattern': pattern, 'name': name}
                    self._set_conf_(name=name, dir_save_conf=saving[1], data=profile_pattern)
                    logging.debug(f'Fichier sauvegardé : {name}')

    def get_profile(self, dir_pattern: str, name_pattern: str):
        """
        Permet de charger le profile_pattern
        """
        profile_pattern = self._set_pattern_(dir_profile_pattern=dir_pattern, name=name_pattern)
        return profile_pattern

    def set_profile(self, time_up: float, time_down: float,
                    pattern: str, name_profile: str,  dir_profile_pattern: str = None, other_path: str = None):
        """
        création du fichier profile pattern qu'il faudra reloader pour l'utiliser.

        Si le dossier "profile_pattern" n'existe pas , il sera créé.

        :param dir_profile_pattern: correspond au nom du fichier (nom du profile)
        :param time_up: correspond au temps de stimulation "A"
        :param time_down: correspond au temps non stimulé "B"
        :param pattern: correspond à la séquence alphabétique du pattern ex: 'CABAB'
        :param name_profile: correspond au nom du fichier (nom du profile)
        :param other_path:

        """
        if not isinstance(other_path, type(None)):
            profile_pattern = self._set_pattern_(other_path=other_path, time_up=time_up,
                           time_down=time_down, pattern=pattern, name=name_profile)
        if isinstance(other_path, type(None)) and not isinstance(dir_profile_pattern, type(None)):
            profile_pattern = self._set_pattern_(dir_profile_pattern=dir_profile_pattern, time_up=time_up,
                           time_down=time_down, pattern=pattern, name=name_profile)


    def transpose_time_pattern(self, reward_time: Series, time_up: float, time_down: float) -> Tuple[Any, ndarray]:
        """
        Cette méthode convertie les intervalles entre les events en séquance alphabétique simple

        :param reward_time: contient les temps en seconde
        :param time_up: temps haut "A"
        :param time_down: temps bas "B"
        """
        reward_time_round = np.around(reward_time.diff(), decimals=4)
        reward_time_round[(reward_time_round != time_up) & (reward_time_round != time_down)] = 'C'
        if time_up == time_down:
            reward_time_round[reward_time_round == time_up] = 'A'
        else:
            reward_time_round[reward_time_round == time_up] = 'A'
            reward_time_round[reward_time_round == time_down] = 'B'
        f = np.array(reward_time_round)
        f = f.sum()
        return f, reward_time_round

    def _detect_structure_pattern_recur_(self, transpos_time: Union[None, ndarray], pattern) -> ndarray:
        """
        cette méthode permet de détecter la structure d'un pattern
        """
        if isinstance(transpos_time, type(None)):
            return pattern.astype(object).sum()
        else:
            if transpos_time[0] == 'A' or transpos_time[0] == 'B':
                pattern = np.hstack((pattern,transpos_time[0]))
                transpos_time = transpos_time[1:]
                return self._detect_structure_pattern_recur_(transpos_time=transpos_time, pattern=pattern)
            elif ['A'] not in transpos_time:
                transpos_time = None
                return self._detect_structure_pattern_recur_(transpos_time=transpos_time, pattern=pattern)

            elif transpos_time[0] == 'C' and len(pattern) < 1:
                transpos_time = transpos_time[1:]
                # print(len(transpos_time))
                return self._detect_structure_pattern_recur_(transpos_time=transpos_time, pattern=pattern)
            elif transpos_time[0] == 'C' and len(pattern) > 1:
                transpos_time = None
                return self._detect_structure_pattern_recur_(transpos_time=transpos_time, pattern=pattern)



    def _pattern_event_(self, reward_time: Series, time_up: float, time_down: float, pattern: str, name: str) -> Tuple[
        Series, ndarray]:
        """
        transformation des intervalle de temps en sequence alphabétique
        afin de trouver le pattern en utilisant les expression recurente
            - en cherchant

        :param reward_time: contient les temps en seconde
        :param time_up: temps haut
        :param time_down: temps bas
        :param pattern: séquence alphabétique correspondant au pattern
        :param name:
        :return: cela donne une serie correspondant au début et fin de toutes les fois ou le pattern est présent
        """

        transpos_time, reward_time_round = self.transpose_time_pattern(reward_time=reward_time, time_up=time_up,
                                                    time_down=time_down)
        pattern2 = np.array([], dtype=int)
        d = np.array(reward_time_round)
        pattern2 = self._detect_structure_pattern_recur_(transpos_time=d, pattern=pattern2)

        iterator_find = re.finditer(pattern, transpos_time)
        z = []
        for match in iterator_find:
            z.append(match.span())
        start_stop = pd.Series(z, name=name)
        time_structure_pattern = reward_time_round
        return start_stop, time_structure_pattern


class EventRewardNeuralynx(ClassificationPattern):
    """
    Cette classe permet de détecter le pattern de synchronisation et le pattern de stimulation
    mais on peut aussi l'implémenter pour détecter d'autres patterns de stimulation.
    la détection ce base sur les intervalles de temps entre les events.
    Pour utiliser cette class :
        - il faut entrer l'ensemble des temps des event issu de "EventFileNeuralynx"
        - if faut entrer les index des temps
    """

    def __init__(self):
        # self.reward_time: Series = all

        self.dir_profile_pattern: str = None #dir_profile_pattern
        self.name_profile_synchro: str = None #name_profile_synchro
        self.name_profile_stimulation: str = None #name_profile_stimulation

        self._pattern_synchro: str = 'CABABABABA'
        self._pattern_stimulation: str = 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
        self.rmz = None
        self.start_stim: Series = None
        self.start_stim_index: Series = None
        self.start_stop_synchro: Series = None
        self.reward_time_ref_rmz: float = None


    @property
    def pattern_synchro(self):
        return self._pattern_synchro

    @pattern_synchro.setter
    def pattern_synchro(self, seq: str):
        self._pattern_synchro = seq

    @property
    def pattern_stimulation(self):
        return self._pattern_stimulation

    @pattern_stimulation.setter
    def pattern_stimulation(self, seq_stim: str):
        self._pattern_stimulation = seq_stim

    def _start_synchro_video_(self, reward_time: Series, profile_pattern: Dict = None) -> Series:
        """
        détection du pattern de synchro video
            -
        :param reward_time:
        :return:
        """
        if isinstance(profile_pattern, type(None)):
            start_stop_synchro, time_structure_pattern = self._pattern_event_(reward_time,
                                    0.0025, 0.0475, self._pattern_synchro, 'synchro_video')
            if start_stop_synchro.size < 1:
                logging.debug(f'Il n"y a pas de ttl de synchronisation')
                return None
            else:
                return start_stop_synchro
        else:
            start_stop_synchro, time_structure_pattern = self._pattern_event_(reward_time,
                                    time_up=profile_pattern['time_up'], time_down=profile_pattern['time_down'],
                                        pattern=profile_pattern['pattern'], name=profile_pattern['name'])
            if start_stop_synchro.size < 1:
                logging.debug(f'Il n"y a pas de ttl de synchronisation (EventTraitement, ligne 274')
                return None
            else:
                return start_stop_synchro


    def _recallage_video_(self, reward_time: Series, profile_pattern: Dict = None) -> Series:
        start_stop_synchro = self._start_synchro_video_(reward_time=reward_time, profile_pattern=profile_pattern)
        return start_stop_synchro

    def _stim_(self, reward_time: Series, profile_pattern: Dict = None) -> Series:
        """
            détection des stims
        :param reward_time: correspond au temps issu des events neuralynx
        :param profile_pattern: profile préenregistré correspondant à une stim
        :return:
        """
        if isinstance(profile_pattern, type(None)):
            start_stop_stim, time_structure_pattern = self._pattern_event_(reward_time,
                                    0.005, 0.005, self._pattern_stimulation, 'stimulation')
            return start_stop_stim
        else:
            start_stop_stim, time_structure_pattern = self._pattern_event_(reward_time,
                                                                          time_up=profile_pattern['time_up'],
                                                                          time_down=profile_pattern['time_down'],
                                                                          pattern=profile_pattern['pattern'],
                                                                          name=profile_pattern['name'])
            return start_stop_stim


    def set_reward(self, reward_time: Series, reward_index: Series,
                   reward_time_from_txt: Series = None, profile_pattern_synchro: Dict = None,
                   profile_pattern_stim: Dict = None, reward_time_ref_rmz: Series=None):
        """
        il faut les temps ainsi que les index
            - donne tous les temps et les index des stimulations

        Si il n'y a pas de TTL de synchro "self.start_stop_synchro" = None
        :param reward_time_ref_rmz: le temps de l'event juste aprés le start de neuralynx moins temps brute en "ms" issue du fichier txt

        :param reward_time: issu de all_event temps en "sec",
                            l'ensemble des event contenu dans le fichier event

        :param reward_index:
        :return:
        """

        self.start_stop_synchro = self._recallage_video_(reward_time=reward_time, profile_pattern=profile_pattern_synchro)
        self.start_stop_stim= self._stim_(reward_time=reward_time, profile_pattern=profile_pattern_stim)
        self.start_stim = pd.Series([reward_time[self.start_stop_stim[i][0]] for i in range(len(self.start_stop_stim))])
        self.start_stim_index = pd.Series([reward_index[self.start_stop_stim[i][0]] for i in range(len(self.start_stop_stim))])
        # print('eee')
        if isinstance(self.start_stop_synchro, type(None)) and not isinstance(reward_time_from_txt, type(None)):

            b = np.around(self.start_stim.diff(), decimals=1)
            a = np.around(reward_time_from_txt.diff() / 1000, decimals=1)
            for i, val in enumerate(b):
                if val == a[1]:
                    print(f'idem {i}: {self.start_stim[i]} : {reward_time_from_txt[1]}')
                    lag = self.start_stim[i] - (reward_time_from_txt[1] / 1000)
                    break
                else:
                    print(f'Aucune correspondance !!!!')
            d = np.around(self.start_stim, decimals=2) - lag
            m = (reward_time_from_txt / 1000) + lag
            self.reward_time_ref_rmz = lag
        else:
            logging.debug(f'Il n"y a pas de ttl de synchronisation (EventTraitement, ligne 274')

            self.reward_time_ref_rmz = reward_time[self.start_stop_synchro[0][0]]

        rmz: Series = reward_time - self.reward_time_ref_rmz
        self.rmz_base_neuralynx = rmz

        return self.start_stim, self.start_stim_index


class EventFileNeuralynx(ExtractEvent, SaveSerialisation):
    """

    """

    def __init__(self, dir_data: str, event: ImportNeuralynx):
        super().__init__()
        self.dir_data: str = dir_data

        self.event: List[DataFrame] = None
        self.info_event: List[DataFrame] = None
        self._event_brute: Event_type = event.Event
        self.name: str = None
        self.path: Path = None

    def __repr__(self):
        return 'EventFileNeuralynx'

    def _recallage_(self, event_time, base_time) -> Series:
        """
        il faut entrer des temps d'events et une base de temps dans laquel chercher

        :return:
        """
        event = pd.Series(self._extraction_event_times_(event_time, base_time), name='event recaller', dtype=int)
        return event

    def _reindex_for_all_event_(self, concat_event: Series, name_series: str) -> Series:
        """
        concat_event est une series
        :param concat_event:
        :return:
        """
        all_event_by_indexing = concat_event.sort_values(ascending=True)
        all_event_by_indexing = all_event_by_indexing.reset_index()
        all_event_by_indexing = all_event_by_indexing.drop(columns='index')
        g = pd.Series(all_event_by_indexing[0][all_event_by_indexing[0].notna()], name=name_series)
        return g

    def _liste_segment_(self, event_brute) -> List[str]:
        """
        retourne une liste unique des segmens présent
        :param event_brute: fichier direct de l'importation
        :return:
        """
        items_segment = list(set([i.split(',')[0] for i in event_brute.keys()]))
        return items_segment

    def _init_final_output_(self) -> defaultdict:
        # items_segment = self._liste_segment_(event_brute)
        final_output = defaultdict(DataFrame)

        return final_output

    def _concate_(self, concat_segment, event_by_segment):
        concat_segment = pd.concat([concat_segment, event_by_segment], ignore_index=True)
        concat_segment = concat_segment[~np.isnan(concat_segment)]
        return concat_segment

    def _decoupe_event_(self, event_brute: Event_type):
        """
        Création des Events recaler sur le temps en seconde
        :return:
        """
        items_segment = self._liste_segment_(event_brute)
        concate_event = self._init_final_output_()
        logging.debug(f'{items_segment}')

        items_event = list(set([i.split(',')[1] for i in event_brute.keys() if i.find('num event') == 17]))
        logging.debug(f'{items_event}')

        info_event = []
        for idx_seg in items_segment:
            concate_segment = pd.DataFrame()
            segment = pd.Series(np.array(event_brute[idx_seg].base_time))

            event_by_segment = pd.DataFrame(columns=items_event)

            concat_time_segment = pd.Series(dtype=float)
            concat_index_segment = pd.Series(dtype=int)

            for i in event_brute.keys():
                if i.find(idx_seg + ',' + ' num event') == 0:
                    idx_event = i.split(',')[1]
                    event_by_segment[idx_event] = pd.Series(np.array(event_brute[idx_seg + ',' + idx_event].event_signals.event_each_times))
                    concat_time_segment = self._concate_(concat_time_segment, event_by_segment[idx_event])

                    logging.debug(f'{idx_seg},{idx_event}')

                    event_by_segment['event recaler'+idx_event] = self._recallage_(event_by_segment[idx_event], segment)
                    concat_index_segment = self._concate_(concat_index_segment, event_by_segment['event recaler'+idx_event])

            concat_time_segment = self._reindex_for_all_event_(concat_time_segment, idx_seg)
            concat_index_segment = self._reindex_for_all_event_(concat_index_segment, idx_seg)

            concate_segment['time'] = concat_time_segment
            concate_segment['index'] = concat_index_segment

            concate_event[idx_seg] = concate_segment

            info_event.append(event_by_segment)

        return info_event, concate_event
        # TODO a finir

    def set_event(self):
        """
        traitement global de la manip avec toutes les segments (session) par neurone.
        :return:
        """
        saving = self._get_conf_(dir_save_conf=self.dir_data, name='all_event')
        if saving[0] == 1:
            info_event, self.event = self._decoupe_event_(self._event_brute)
            all_event = {'info_event': info_event, 'all_event': self.event}
            self._set_conf_(name='all_event', dir_save_conf=saving[1], data=all_event)
            return info_event, self.event

        else:
            logging.debug(f'chargement du fichier : all_event')
            all_event = self._chargement_fichier_data_(path=saving[1], name='all_event')
            info_event = all_event['info_event']
            self.event = all_event['all_event']

            return info_event, self.event

        # TODO a finir


if __name__ == "__main__":
    t1 = chrono.time()
    # csc_dir: str = r'D:\Dropbox\python\import_neuralynxv2\data\pinp8 06022020'
    # csc_dir: str = r'D:\Dropbox\python\import_neuralynxv2\data\pinp2_bsl 16012020'

    dir_neuralynx: str = r'/data/cplx07 + bsl'
    # csc_dir: str = r'D:\Dropbox\python\import_neuralynxv2\data\equequ1 - test Steve'
    event_brute = LoadData.init_data(ImportNeuralynx, dir_neuralynx, 'event')


    Event = EventFileNeuralynx(dir_data=dir_neuralynx, event=event_brute)
    info_event, all_event = Event.set_event()

    reward = EventRewardNeuralynx()

    start_stim, start_stim_index = reward.set_reward(reward_time=all_event['num segment : 1']['time'], reward_index=all_event['num segment : 1']['index'])

    omission_recallage = EventFromOther()
    omission_recallage.start()

    t2 = chrono.time()

    print(f'temps global : {t2 - t1}')
