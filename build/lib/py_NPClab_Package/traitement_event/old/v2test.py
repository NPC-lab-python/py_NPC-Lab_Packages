import logging
import bisect
from quantities.quantity import Quantity
import numpy as np
from numpy.core.multiarray import ndarray
from pathlib import Path
from typing import List, Dict, Union
from load.old.Load_neuralynx_separate import GlobalEventBasetime, GlobalEventSignal
from NPClab_Package.utilitaire_load import LoadData, ImportNeuralynx

import time as chrono

import pandas as pd
from pandas import DataFrame, Series

Event_type = Dict[str, Union[GlobalEventSignal, GlobalEventBasetime]]
logging.basicConfig(level=logging.DEBUG)
# logging.debug('tesst')
# logging.disable(logging.CRITICAL)

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
        bin = np.arange(0, longueur, self.bin_decoupage)
        der_bin = self.bin_decoupage-(longueur-bin[-1])
        bin = np.arange(0, longueur+der_bin+1, self.bin_decoupage)

        base_index = np.arange(0, bin[-1], 1)
        event_time = event_time[~np.isnan(event_time)]

        vector_index = np.zeros((1, self.bin_decoupage), dtype=int)
        vector_time = np.zeros((1, self.bin_decoupage), dtype=float)
        vector_event = []

        time = base_time
        time = np.append(np.array(time), np.zeros(der_bin))
        time[len(base_time):] = time[len(base_time) - 1]

        for i in range(len(bin)-1):
            vector_index = np.vstack((vector_index, np.zeros((1, self.bin_decoupage), dtype=int)))
            vector_time = np.vstack((vector_time, np.zeros((1, self.bin_decoupage), dtype=float)))
            vector_index[i] = base_index[bin[i]:bin[i+1]]
            vector_time[i] = time[bin[i]:bin[i+1]]

            # logging.debug(f'borne vector_time :{i} {vector_time[i].min()} {vector_time[i].max()}')

        for i in range(len(bin)):
            vector_event.append([self._tri_event_time_(event_time, vector_time[i])])

        return vector_time, vector_event, vector_index

    def _tri_event_time_(self,event_time: Quantity, vector_time: ndarray):
        mi = event_time >= vector_time.min()
        ma = event_time <= vector_time.max()
        uu = mi & ma
        tmp = event_time[uu]
        return tmp

    def _extraction_event_times_(self, event_time: Quantity, base_time: Quantity) -> ndarray:
        """
        il faut entrer des temps d'event et une base de temps
        :param event_time:
        :param base_time:
        :return:
        """
        vector_time, vector_event, vector_index = self.__decoupe_event__(event_time, base_time)

        logging.debug(f'Début de l"extraction des times des events avec bisect')
        logging.debug(f'Nombre d"event" : {len(event_time)}')

        event_index_in_raw_bisect = self._extraction_(vector_time, vector_event, vector_index)
        return event_index_in_raw_bisect
        # TODO a finir

    def _extraction_(self, vector_time, vector_event, vector_index) -> ndarray:
        event_index_in_raw_bisect: ndarray = np.array([], dtype=int)
        s3 = chrono.time()
        for n in range(len(vector_time)):
            # logging.debug(f'vector_time : {n}')
            vector = vector_event[n]
            if len(vector[0]) == 0:
                pass
            else:
                for idx in range(len(vector[0])):
                    # logging.debug(f'vector_event : {vector[0].iloc[idx]}')
                    event_index_in_raw_bisect = np.hstack(
                        (
                        event_index_in_raw_bisect, vector_index[n][bisect.bisect(vector_time[n], vector[0].iloc[idx])]))

        s4 = chrono.time()
        logging.debug(f'Fin Event, durée avec bisect : {s4 - s3}')
        return event_index_in_raw_bisect


class EventFileNeuralynx(ExtractEvent):
    """

    """

    def __init__(self, event: ImportNeuralynx):
        super().__init__()
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

    def _decoupe_event_par_segment_(self, event_brute: Event_type):
        """
        Création des Events recaler sur le temps en seconde
        :return:
        """
        items_segment = list(set([i.split(',')[0] for i in event_brute.keys()]))
        logging.debug(f'{items_segment}')

        items_event = list(set([i.split(',')[1] for i in event_brute.keys() if i.find('num event') == 17]))
        logging.debug(f'{items_event}')

        info_event = []
        concate_event = []
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

                    concat_time_segment = pd.concat([concat_time_segment, event_by_segment[idx_event]], ignore_index=True)
                    concat_time_segment = concat_time_segment[~np.isnan(concat_time_segment)]

                    logging.debug(f'{idx_seg},{idx_event}')

                    event_by_segment['event recaler'+idx_event] = self._recallage_(event_by_segment[idx_event], segment)

                    concat_index_segment = pd.concat([concat_index_segment, event_by_segment['event recaler'+idx_event]], ignore_index=True)
                    concat_index_segment = concat_index_segment[~np.isnan(concat_index_segment)]

                    concat_index_segment = concat_index_segment.astype('int64')

            concat_time_segment = self._reindex_for_all_event_(concat_time_segment, idx_seg)
            concat_index_segment = self._reindex_for_all_event_(concat_index_segment, idx_seg)

            concate_segment[idx_seg+' time'] = concat_time_segment
            concate_segment[idx_seg+' index'] = concat_index_segment

            concate_event.append(concate_segment)

            info_event.append(event_by_segment)

        return info_event, concate_event
        # TODO a finir



    def make_event(self):
        """
        traitement global de la manip avec toutes les segments (session) par neurone.
        :return:
        """
        info_event, self.event = self._decoupe_event_par_segment_(self._event_brute)
        return info_event, self.event
        # TODO a finir



if __name__ == "__main__":
    t1 = chrono.time()
    # csc_dir: str = r'D:\Dropbox\python\import_neuralynxv2\data\pinp8 06022020'
    # csc_dir: str = r'D:\Dropbox\python\import_neuralynxv2\data\pinp2_bsl 16012020'

    dir_neuralynx: str = r'/data/cplx07 + bsl'
    # csc_dir: str = r'D:\Dropbox\python\import_neuralynxv2\data\equequ1 - test Steve'
    event_brute = LoadData.init_data(ImportNeuralynx, dir_neuralynx, 'event')
    Event = EventFileNeuralynx(event=event_brute)
    info_event, all_event = Event.make_event()

    t2 = chrono.time()

    print(f'temps global : {t2 - t1}')