from quantities.quantity import Quantity
import numpy as np
from numpy.core.multiarray import ndarray
from pathlib import Path
from typing import List, Dict, Union, Tuple
from load.old.Load_neuralynx_separate import GlobalEventBasetime, GlobalEventSignal
from NPClab_Package.utilitaire_load import LoadData, ImportNeuralynx

import time as chrono

import pandas as pd
from pandas import Series
import re

Event_type = Dict[str, Union[GlobalEventSignal, GlobalEventBasetime]]



class ExtractEvent(object):
    """
    Permet de créer les Events sur la meme base de temps que le temps du signal brute (rawsignal)
    """

    def __init__(self):
        self._bin_decoupage: int = 512000
        self._sample: int = 32000

    @property
    def bin_decoupage(self):
        return self._bin_decoupage

    @bin_decoupage.setter
    def bin_decoupage(self, val: int):
        self._bin_decoupage = val

    @property
    def sample(self):
        return self._sample

    @sample.setter
    def sample(self, val: int):
        self._sample = val

    def __decoupe_event__(self, event_time: Quantity, base_time: Quantity):
        longueur = base_time.size
        bin = list(range(0, longueur, self.bin_decoupage))
        base_index = list(range(0, longueur))
        if bin[-1] < longueur:
            bin.append(longueur - bin[-1] + bin[-1])
        vector_time = []
        vector_event = []
        time = base_time
        for i in range(len(bin) - 1):
            vector_time.append([base_index[bin[i]:bin[i + 1]], time[bin[i]:bin[i + 1]]])
            vector_event.append(
                event_time[(event_time > time[bin[i]]) & (event_time < time[bin[i + 1] - 1])])
        return vector_time, vector_event

    def _extraction_event_times_(self, event_time: Quantity, base_time: Quantity) -> ndarray:
        """
        il faut entrer des temps d'event et une base de temps
        :param event_time:
        :param base_time:
        :return:
        """
        event_time_in_raw: ndarray([int]) = np.array([], dtype=int)
        vector_event_time: ndarray([float])
        vector_time, vector_event_time = self.__decoupe_event__(event_time, base_time)
        print(f'Début de l"extraction des times des events')
        print(f'Nombre d"event" : {len(event_time)}')

        s1 = chrono.time()
        for n in range(len(vector_time)):
            if vector_event_time[n].size == 0:
                pass
            else:
                for i in range(len(vector_event_time[n])):
                    posi_temp = np.where(
                        (vector_time[n][1] > float(vector_event_time[n][i]) - 1 / self.sample) & (
                                vector_time[n][1] < float(vector_event_time[n][i]) + 1 / self.sample))
                    event_time_in_raw = np.hstack((event_time_in_raw, vector_time[n][0][posi_temp[0][0]]))
        s2 = chrono.time()
        print(f'Fin de l"extraction des times des Event, durée : {s2 - s1}')
        return event_time_in_raw


class EventFromOther(ExtractEvent):
    """
    Cette class permet d'obtenir les valeurs d'event provenant d'une autre origine que neuralynx
    comme les
    """

    def __init__(self):
        super().__init__()
        # self.event_time: ndarray = event_time
        # self.base_time: Quantity = base_time
        self.event_time_in_raw: ndarray([int]) = np.array([], dtype=int)

    def _pre_format_(self, event_time):
        event_time = Quantity(event_time/1000, 's')
        return event_time

    def _rmz_base_time_(self, time_event_ref_rmz: float, base_time: Quantity):
        base_time = base_time - time_event_ref_rmz
        return base_time

    def start(self, event_time: ndarray, base_time: Quantity, time_event_ref_rmz: float):
        time = self._pre_format_(event_time)
        base_time = self._rmz_base_time_(time_event_ref_rmz, base_time)
        self.event_time_in_raw = self._extraction_event_times_(time, base_time)


class EventRewardNeuralynx(object):
    """
    Cette classe permet de détecter le pattern de synchronisation et le pattern de stimulation
    mais on peut aussi l'implémenter pour détecter d'autres patterns de stimulation
    """

    def __init__(self):
        # self.reward_time: Series = all
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

    def _start_indexing_(self, reward_time: Series) -> Series:
        start_stop_synchro = self._pattern_event_(reward_time, 0.0025, 0.0475, self._pattern_synchro, 'synchro_video')
        return start_stop_synchro

    def _pattern_event_(self, reward_time: Series, time_up: float, time_down: float, pattern: str, name: str) -> Series:
        """

        :param reward_time: contient les temps en seconde
        :param time_up:
        :param time_down:
        :param pattern:
        :param name:
        :return:
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
        iterator_find = re.finditer(pattern, f)
        z = []
        for match in iterator_find:
            z.append(match.span())
        start_stop = pd.Series(z, name=name)
        return start_stop

    def _rmz_(self, reward_time: Series) -> Series:
        start_stop_synchro = self._start_indexing_(reward_time)
        self.start_stop_synchro = start_stop_synchro
        self.reward_time_ref_rmz = reward_time[start_stop_synchro[0][0]]
        rmz: Series = reward_time - self.reward_time_ref_rmz
        return rmz

    def _stim_(self, rmz: Series, reward_time: Series, reward_index: Series) -> Tuple[Series, Series]:
        start_stop_stim = self._pattern_event_(reward_time, 0.005, 0.005, self._pattern_stimulation,
                                               'stimulation')
        start_stim = pd.Series([rmz[start_stop_stim[i][0]] for i in range(len(start_stop_stim))])
        start_stim_index = pd.Series([reward_index[start_stop_stim[i][0]] for i in range(len(start_stop_stim))])
        return start_stim, start_stim_index

    def make_reward(self, reward_time: Series, reward_index: Series):
        self.rmz = self._rmz_(reward_time)
        self.start_stim, self.start_stim_index = self._stim_(self.rmz, reward_time, reward_index)
        return self.start_stim, self.start_stim_index


class EventFileNeuralynx(ExtractEvent):
    """

    """

    def __init__(self, event: ImportNeuralynx):
        super().__init__()
        self.event: Dict[str, Dict[str]] = {}

        self._event_segment: Dict[str, Dict[str, Union[Quantity, ndarray]]] = {}
        self._event_segment_on: Dict[str, Dict[str, Union[Quantity, ndarray]]] = {}
        self._event_brute: Event_type = event.Event
        self.name: str = None
        self.path: Path = None
        # self.reward: Series = None

    def __repr__(self):
        return 'EventFileNeuralynx'

    def _recallage_(self, event_each_times: Quantity, base_time: Quantity, event: Dict[str, Dict[str, Union[Quantity, ndarray]]], segment: str ):
        """
        il faut entrer des temps d'events et une base de temps dans laquel chercher
        :param event_each_times:
        :param base_time:
        :param event:
        :param segment:
        :return:
        """

        event[segment] = {'event brute': event_each_times}
        event[segment].update({'event recaler': self._extraction_event_times_(event_each_times, base_time)})
        return event

    def _recallage_by_segment_(self, event_brute: Event_type,
                               _event_segment: Dict[str, Dict[str, Union[Quantity, ndarray]]],
                               _event_segment_on: Dict[str, Dict[str, Union[Quantity, ndarray]]]):

        _tmp_items_name_event = [i for i in event_brute.keys() if i.find('num event') != -1]

        for idx, segment in enumerate(_tmp_items_name_event):

            if segment.find('event start') != -1:
                print(f'Event start : {segment}')

                _event_segment = self._recallage_(event_brute[segment].event_signals.event_each_times,
                                                  event_brute[segment.split(',')[0]].base_time, _event_segment, segment)

            elif segment.find('event start') == -1:
                print(f'Event on : {segment} ')

                _event_segment_on = self._recallage_(event_brute[segment].event_signals.event_each_times,
                                                     event_brute[segment.split(',')[0]].base_time, _event_segment_on, segment)

        return _event_segment, _event_segment_on

    def _decoupe_event_par_segment_(self, event_brute: Event_type,
                                    _event_segment: Dict[str, Dict[str, Union[Quantity, ndarray]]],
                                    _event_segment_on: Dict[str, Dict[str, Union[Quantity, ndarray]]]) -> Tuple[
        Dict, Dict, Dict]:
        """
        Création des Events recaler sur le temps en seconde
        :return:
        """
        self.all: Dict[str, List[Series]] = {}

        all: Dict[str, Series] = {}
        _event_segment, _event_segment_on = self._recallage_by_segment_(event_brute, _event_segment, _event_segment_on)

        _tmp_items_name_event = [i for i in event_brute.keys() if i.find('num event') != -1]

        for idx, segment in enumerate(_tmp_items_name_event):
            all = self._concate_in_segment_(all, segment, _event_segment)
            all = self._concate_in_segment_(all, segment, _event_segment_on)

            print(f'segment: {segment}')
            all_event_by_time, all_event_by_indexing = self._recal_all_event_(all)
            self.all[segment] = [all_event_by_time, all_event_by_indexing]
            
        return _event_segment, _event_segment_on, self.all
        # TODO a finir

    def _pre_format_(self, _tmp_items_segment: list,
                     event_segment: Dict[str, Dict[str, Union[Quantity, ndarray]]], event):
        """

        :param _tmp_items_segment:
        :param event_segment:
        :param event:
        :return:
        """
        for idx, items in enumerate(event_segment.keys()):
            for i in _tmp_items_segment:
                if items.split(',')[0] == i:
                    print(i, items)
                    event[i].update({items: event_segment[items]})
        return event

    def _pre_format_par_segment_(self, _event_segment: Dict[str, Dict[str, Union[Quantity, ndarray]]],
                                 _event_segment_on: Dict[str, Dict[str, Union[Quantity, ndarray]]],
                                 event_brute: Event_type):
        """
        recupération de la construction de decoupe_event_par_segment
        :return:
        """
        _tmp_items = set([i.split(',')[0] for i in event_brute.keys()])
        _tmp_items_segment = [i for i in _tmp_items]
        event = {x: {} for x in _tmp_items_segment}
        event = self._pre_format_(_tmp_items_segment, _event_segment, event)
        event = self._pre_format_(_tmp_items_segment, _event_segment_on, event)
        return event

    def _reward_(self, event_brut: ndarray):
        """
        recupération des valeurs en temps sur les quels on fait la différences (on garde que les diff de plus de 50ms)
        :param event_brut: Event.event['num segment : 1']['num segment : 1, num event start : 0']['event brute']
        :return:
        """
        seuil = 0
        _tmp_ = pd.Series(event_brut)
        _tmp_intervalle: Series = _tmp_.diff(1)
        _tmp_reward_ = pd.Series(_tmp_intervalle.index[_tmp_intervalle > seuil])
        return _tmp_reward_

    def _concate_in_segment_(self, all: Dict[str, Series], segment: str,
                             _event_segment: Dict[str, Dict[str, Union[Quantity, ndarray]]]) -> Dict[
        str, Series]:
        """
        permet de regroupe dans un dict tout les event de segment ayant plus d'un start stop comme event
        :param all:
        :param segment:
        :param _event_segment:
        :return:
        """
        if segment in _event_segment:
            if len(_event_segment[segment]['event brute']) > 2:
                all.update({segment: pd.Series(_event_segment[segment]['event brute'], name='brute')})
                all.update(
                    {segment + 'recaler': pd.Series(_event_segment[segment]['event recaler'], name='recaler')})
        return all

    def _recal_all_event_(self, all: Dict) -> Tuple[Series, Series]:
        a = pd.DataFrame(all)

        items = list(a.columns)

        b = [a[items[i]] for i in range(len(items)) if items[i].find('recaler') == -1]
        all_event_by_time = self._reindex_for_all_event_(b, 'event_time')

        e = [a[items[i]] for i in range(len(items)) if items[i].find('recaler') != -1]
        all_event_by_indexing = self._reindex_for_all_event_(e, 'event_time_recaler')

        return all_event_by_time, all_event_by_indexing

    def _reindex_for_all_event_(self, e: List[Series], name_series: str) -> Series:
        """
        e est une list contenant la series
        :param e:
        :return:
        """
        all_event_by_indexing = pd.Series(dtype=float)

        for i in range(len(e)):
            all_event_by_indexing = all_event_by_indexing.append(e[i])
        all_event_by_indexing = all_event_by_indexing.sort_values(ascending=True)
        all_event_by_indexing = all_event_by_indexing.reset_index()
        all_event_by_indexing = all_event_by_indexing.drop(columns='index')
        g = pd.Series(all_event_by_indexing[0][all_event_by_indexing[0].notna()], name=name_series)
        return g

    def make_event(self):
        """
        traitement global de la manip avec toutes les segments (session) par neurone.
        :return:
        """
        self._event_segment, self._event_segment_on, self.all = self._decoupe_event_par_segment_(self._event_brute,
                                                                                                 self._event_segment,
                                                                                                 self._event_segment_on)

        self.event = self._pre_format_par_segment_(self._event_segment, self._event_segment_on, self._event_brute)
        return self.all


    # @property
    # def reward(self):
    #     return


if __name__ == "__main__":
    t1 = chrono.time()
    # csc_dir: str = r'D:\Dropbox\python\import_neuralynxv2\data\pinp8 06022020'
    # csc_dir: str = r'D:\Dropbox\python\import_neuralynxv2\data\pinp2_bsl 16012020'

    dir_neuralynx: str = r'/data/cplx07 + bsl'
    # csc_dir: str = r'D:\Dropbox\python\import_neuralynxv2\data\equequ1 - test Steve'
    event_brute = LoadData.init_data(ImportNeuralynx, dir_neuralynx, 'event')
    Event = EventFileNeuralynx(event=event_brute)
    all = Event.make_event()
    EventRewardNeuralynx(all=all)
    t2 = chrono.time()

    print(f'temps global : {t2 - t1}')
