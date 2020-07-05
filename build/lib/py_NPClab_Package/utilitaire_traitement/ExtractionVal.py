import logging
import bisect
from quantities.quantity import Quantity
import numpy as np
from numpy.core.multiarray import ndarray
from typing import List, Dict, Union, Tuple
from py_NPClab_Package.utilitaire_neuralynx.Load_neuralynx import GlobalEventBasetime, GlobalEventSignal

import concurrent
import time as chrono

from pandas import Series

Event_type = Dict[str, Union[GlobalEventSignal, GlobalEventBasetime]]

logging.basicConfig(level=logging.DEBUG)
# logging.disable(logging.CRITICAL)


class ExtractEvent(object):
    """
    VERSION MULTIPROCESS MAP
    Permet de créer les Events sur la meme base de temps que le temps du signal brute (rawsignal)
    Pour pouvoir utiliser cette class :
        - entrer les temps sous forme de series
        - entrer la base de temps dans laquel on veut extraire les temps

    """

    def __init__(self):
        self._bin_decoupage: int = 512000 #1024000#
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

    def __decoupe_event__(self, event_time: Series, base_time: Quantity) -> ndarray:
        """
        permet la création
            - d'un vecteur index constitué d'une matrice numpy contenant l'index de tout les temps
            - d'un vecteur time constitué d'une matrice numpy contenant tout les temps corredspiondant a l'index
            - d'une liste de serie contenant les temps entrer découpé pour chaque bien

        ce découpage permet une recherche plus rapide des valeurs

        :param event_time:
        :param base_time:
        :return:
        """
        longueur = base_time.size
        bin = np.arange(0, longueur, self.bin_decoupage)
        der_bin = self.bin_decoupage-(longueur-bin[-1])
        bin = np.arange(0, longueur+der_bin+1, self.bin_decoupage)

        event_time = event_time[~np.isnan(event_time)]
        base_time = np.array(base_time)
        time = np.append(base_time, np.zeros(der_bin))
        time[len(base_time):] = time[len(base_time) - 1]


        event_index_in_raw: ndarray = np.array([], dtype=int)
        numbers = list(range(len(bin)-1))
        argument = [[event_time, time, bin, n] for n in numbers]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(self._extractionV2_, argument)
            for f in results:
                event_index_in_raw = np.hstack((event_index_in_raw, f))

        # for n in range(len(bin)-1):
        #     event_index_in_raw = np.hstack((event_index_in_raw, self._extraction_(event_time, time, bin, n)))

        event_index_in_raw.sort()
        return event_index_in_raw

    def _tri_event_time_(self, event_time: Series, vector_time: ndarray) -> Series:
        """
        donne toutes les valeurs comprises dans l'intervalle du bin
        :param event_time:
        :param vector_time: bin de temps
        :return:
        """
        mi = event_time >= vector_time.min()
        ma = event_time <= vector_time.max()
        uu = mi & ma
        tmp = event_time[uu]
        return tmp

    def _extraction_event_times_(self, event_time: Series, base_time: Union[Series, Quantity]) -> ndarray:
        """
        il faut entrer des temps d'event et une base de temps
        :param event_time: il est préferable que cela soit une pd.Series
        :param base_time:
        :return:
        """
        logging.debug(f'Début de l"extraction')
        logging.debug(f'Nombre d"event" : {len(event_time)}')
        event_index_in_raw_bisect = self.__decoupe_event__(event_time, base_time)
        logging.debug(f'fin de l"extraction')
        return event_index_in_raw_bisect
        # TODO a finir

    def _extraction_(self, event_time, time, bin, n) -> ndarray:
        event_index_in_raw_bisect: ndarray = np.array([], dtype=int)
        vector = self._tri_event_time_(event_time, time[bin[n]:bin[n + 1]])
        vect_index = np.arange(bin[n], bin[n + 1], 1)
        vect_time = time[bin[n]:bin[n + 1]]

        if len(vector) == 0:
            pass
        else:
            for idx in range(len(vector)):
                event_index_in_raw_bisect = np.hstack((event_index_in_raw_bisect, self._extra_val_(vect_index, vect_time, vector.iloc[idx])))
        return event_index_in_raw_bisect

    def _extractionV2_(self, *args) -> ndarray:
        event_time = args[0][0]
        time = args[0][1]
        bin = args[0][2]
        n = args[0][3]
        event_index_in_raw_bisect: ndarray = np.array([], dtype=int)
        vector = self._tri_event_time_(event_time, time[bin[n]:bin[n + 1]])
        vect_index = np.arange(bin[n], bin[n + 1], 1)
        vect_time = time[bin[n]:bin[n + 1]]

        if len(vector) == 0:
            pass
        else:
            for idx in range(len(vector)):
                event_index_in_raw_bisect = np.hstack((event_index_in_raw_bisect, self._extra_val_(vect_index, vect_time, vector.iloc[idx])))
        return event_index_in_raw_bisect

    def _extra_val_(self, vect_index, vect_time, val):
        tmp = vect_index[bisect.bisect(vect_time, val)]
        return tmp

class ExtractEventoptimise(object):
    """
    VERSION MULTIPROCESS JUSTE OPTIMISER
    Permet de créer les Events sur la meme base de temps que le temps du signal brute (rawsignal)
    Pour pouvoir utiliser cette class :
        - entrer les temps sous forme de series
        - entrer la base de temps dans laquel on veut extraire les temps

    """

    def __init__(self):
        self._bin_decoupage: int = 1024000#512000
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

    def __decoupe_event__(self, event_time: Series, base_time: Quantity) -> ndarray:
        """
        permet la création
            - d'un vecteur index constitué d'une matrice numpy contenant l'index de tout les temps
            - d'un vecteur time constitué d'une matrice numpy contenant tout les temps corredspiondant a l'index
            - d'une liste de serie contenant les temps entrer découpé pour chaque bien

        ce découpage permet une recherche plus rapide des valeurs

        :param event_time:
        :param base_time:
        :return:
        """
        t1 = chrono.perf_counter()

        longueur = base_time.size
        bin = np.arange(0, longueur, self.bin_decoupage)
        der_bin = self.bin_decoupage-(longueur-bin[-1])
        bin = np.arange(0, longueur+der_bin+1, self.bin_decoupage)

        event_time = event_time[~np.isnan(event_time)]

        time = np.append(np.array(base_time), np.zeros(der_bin))
        time[len(base_time):] = time[len(base_time) - 1]

        t2 = chrono.perf_counter()
        print(f'temps preformat avant bisect : {t2 - t1}')
        s3 = chrono.time()
        event_index_in_raw: ndarray = np.array([], dtype=int)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = [executor.submit(self._extraction_, event_time, time, bin, n) for n in range(len(bin)-1)]
            for f in concurrent.futures.as_completed(results):
                event_index_in_raw = np.hstack((event_index_in_raw, f.result()))

        # for n in range(len(bin)-1):
        #     event_index_in_raw = np.hstack((event_index_in_raw, self._extraction_(event_time, time, bin, n)))
        event_index_in_raw.sort()
        s4 = chrono.time()
        logging.debug(f'Fin Event, durée all : {s4 - s3}')
        return event_index_in_raw

    def _tri_event_time_(self, event_time: Series, vector_time: ndarray) -> Series:
        """
        donne toutes les valeurs comprises dans l'intervalle du bin
        :param event_time:
        :param vector_time: bin de temps
        :return:
        """
        mi = event_time >= vector_time.min()
        ma = event_time <= vector_time.max()
        uu = mi & ma
        tmp = event_time[uu]
        return tmp

    def _extraction_event_times_(self, event_time: Series, base_time: Quantity) -> ndarray:
        """
        il faut entrer des temps d'event et une base de temps
        :param event_time: il est préferable que cela soit une pd.Series
        :param base_time:
        :return:
        """
        logging.debug(f'Début de l"extraction')
        logging.debug(f'Nombre d"event" : {len(event_time)}')
        event_index_in_raw_bisect = self.__decoupe_event__(event_time, base_time)
        logging.debug(f'fin de l"extraction')
        return event_index_in_raw_bisect
        # TODO a finir

    def _extraction_(self, event_time, time, bin, n) -> ndarray:
        event_index_in_raw_bisect: ndarray = np.array([], dtype=int)
        vector = self._tri_event_time_(event_time, time[bin[n]:bin[n + 1]])
        vect_index = np.arange(bin[n], bin[n + 1], 1)
        vect_time = time[bin[n]:bin[n + 1]]

        if len(vector) == 0:
            pass
        else:
            for idx in range(len(vector)):
                event_index_in_raw_bisect = np.hstack((event_index_in_raw_bisect, self._extra_val_(vect_index, vect_time, vector.iloc[idx])))
        return event_index_in_raw_bisect

    def _extra_val_(self, vect_index, vect_time, val):
        tmp = vect_index[bisect.bisect(vect_time, val)]
        return tmp


class ExtractEventold(object):
    """
        ANCIENNE VERSION BEAUCOUP PLUS LENTE
    Permet de créer les Events sur la meme base de temps que le temps du signal brute (rawsignal)
    Pour pouvoir utiliser cette class :
        - entrer les temps sous forme de series
        - entrer la base de temps dans laquel on veut extraire les temps

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

    def __decoupe_event__(self, event_time: Series, base_time: Quantity) -> Tuple[ndarray, List[List[Series]], ndarray]:
        """
        permet la création
            - d'un vecteur index constitué d'une matrice numpy contenant l'index de tout les temps
            - d'un vecteur time constitué d'une matrice numpy contenant tout les temps corredspiondant a l'index
            - d'une liste de serie contenant les temps entrer découpé pour chaque bien

        ce découpage permet une recherche plus rapide des valeurs

        :param event_time:
        :param base_time:
        :return:
        """
        t1 = chrono.perf_counter()

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
        t2 = chrono.perf_counter()

        print(f'temps preformat avant bisect : {t2 - t1}')
        return vector_time, vector_event, vector_index

    def _tri_event_time_(self, event_time: Series, vector_time: ndarray) -> Series:
        """
        donne toutes les valeurs comprises dans l'intervalle du bin
        :param event_time:
        :param vector_time: bin de temps
        :return:
        """
        mi = event_time >= vector_time.min()
        ma = event_time <= vector_time.max()
        uu = mi & ma
        tmp = event_time[uu]
        return tmp

    def _extraction_event_times_(self, event_time: Series, base_time: Quantity) -> ndarray:
        """
        il faut entrer des temps d'event et une base de temps
        :param event_time: il est préferable que cela soit une pd.Series
        :param base_time:
        :return:
        """

        vector_time, vector_event, vector_index = self.__decoupe_event__(event_time, base_time)

        logging.debug(f'Début de l"extraction des times des events avec bisect')
        logging.debug(f'Nombre d"event" : {len(event_time)}')

        event_index_in_raw_bisect = self._extraction_(vector_time, vector_event, vector_index)
        return event_index_in_raw_bisect
        # TODO a finir

    def _extraction_(self, vector_time: ndarray, vector_event: List[List[Series]], vector_index) -> ndarray:
        s3 = chrono.time()
        event_index_in_raw_bisect: ndarray = np.array([], dtype=int)
        for n in range(len(vector_time)):
            # logging.debug(f'vector_time : {n}')
            vector = vector_event[n]
            if len(vector[0]) == 0:
                pass
            else:
                for idx in range(len(vector[0])):
                    # logging.debug(f'vector_event : {vector[0].iloc[idx]}')
                    event_index_in_raw_bisect = np.hstack((event_index_in_raw_bisect,
                        vector_index[n][bisect.bisect(vector_time[n], vector[0].iloc[idx])]))

        s4 = chrono.time()
        logging.debug(f'Fin Event, durée avec bisect : {s4 - s3}')
        return event_index_in_raw_bisect

# if __name__ == "__main__":
#     t1 = chrono.perf_counter()
#     dir_save = r'/data/cplx07 + bsl/save'
#
#     name1 = 'segment1_neurone0'
#     neurone = LoadData.init_data(NeuroneFilesSerialiser, dir_save, name1)
#
#     name2 = 'segment_infos'
#     segment_infos = LoadData.init_data(SegmentFilesSerialiser, dir_save, name2, 1)
#
#     event = ExtractEvent()
#     event_index_in_raw_bisect = event._extraction_event_times_(event_time=neurone.data, base_time=segment_infos.data)
#     t2 = chrono.perf_counter()
#
#     print(f'temps global : {t2 - t1}')
#     print('fin')
