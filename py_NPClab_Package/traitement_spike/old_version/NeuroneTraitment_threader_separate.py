import numpy as np
from numpy.core.multiarray import ndarray
from pathlib import Path
import glob
from typing import List, Dict
from load.old.Load_neuralynx_separate import RawSignals
import time as chrono
from threading import Thread
import threading as thread
import logging
logging.basicConfig(level=logging.DEBUG)



class ImportRawSignal(RawSignals):
    def __init__(self, csc_dir: str):
        super().__init__(csc_dir)
        self.load(option='csc')


class ExtractSpike(object):

    def __decoupe_time__(self, spike_time, basetime: ndarray):
        """
        entrer la base temporel issu de signal_segment pour chaque segment
        :param spike_time:
        :param signal:
        :return:
        """
        longueur = basetime.size
        bin = list(range(0, longueur, 512000))
        base_index = list(range(0, longueur))
        if bin[-1] < longueur:
            bin.append(longueur - bin[-1] + bin[-1])
        vector_time = []
        vector_spike_time = []
        time = basetime
        for i in range(len(bin) - 1):
            vector_time.append([base_index[bin[i]:bin[i + 1]], time[bin[i]:bin[i + 1]]])
            vector_spike_time.append(
                spike_time[(spike_time > time[bin[i]]) & (spike_time < time[bin[i + 1] - 1])])
        return vector_time, vector_spike_time

    def _extraction_times_(self, spike_time, basetime: ndarray):
        spike_time_in_raw: ndarray([int]) = np.array([], dtype=int)
        vector_time, vector_spike_time = self.__decoupe_time__(spike_time, basetime)
        print(f'Début de l"extraction des times des spikes')
        print(f'Nombre de spike : {len(spike_time)}')
        s1 = chrono.time()
        for n in range(len(vector_time)):
            for i in range(len(vector_spike_time[n])):
                posi_temp = np.where((vector_time[n][1] > vector_spike_time[n][i] - 1 / 32000) & (
                        vector_time[n][1] < vector_spike_time[n][i] + 1 / 32000))
                spike_time_in_raw = np.hstack((spike_time_in_raw, vector_time[n][0][posi_temp[0][0]]))
        s2 = chrono.time()
        print(f'Fin de l"extraction des times des spikes, durée : {s2 - s1}')
        return spike_time_in_raw

    def start(self, spike_time):
        self.spike_time_in_raw = self._extraction_times_(spike_time, self.basetime)
        # TODO : identifier les besoins d'extraction des waveform


class SpikeFiles(ExtractSpike):

    def __init__(self, spike_times_rmz: ndarray, basetime: ndarray):
        super().__init__()
        self.basetime: ndarray = basetime
        self._spike_times_rmz: ndarray([float]) = spike_times_rmz[0]
        self._spike_times_isi: ndarray([float]) = np.array([], dtype=float)
        self.spike_times_isi: ndarray([float]) = None

        self._spike_swb_percent: float = None
        self.all_info_swb: Dict[str, ndarray] = {}

    def __repr__(self):
        return 'SpikeFiles'

    def __interspikeinterval__(self):
        self._spike_times_isi = self._spike_times_rmz[1:] - self._spike_times_rmz[:-1]

    def _spikeinburst_(self):
        on_swb: int = 0
        off_swb: int = 0
        taille_max: int = self._spike_times_isi[:-1].size

        ISI = self._spike_times_isi
        spike_swb_start = np.full([taille_max], False)
        spike_swb_stop = np.full([taille_max], False)
        spike_swb_all = np.full([taille_max], False)

        all_info_swb: Dict[str, ndarray] = {}
        nb_spike_in_burst = np.array([], dtype=int)
        val_spike_in_burst: List = []

        idx_swb_start = np.array([], dtype=int)
        idx_swb_stop = np.array([], dtype=int)

        while on_swb < taille_max:
            idx_swb = []

            if ISI[on_swb] < 0.08:
                idx_swb_start = np.hstack((idx_swb_start, on_swb))
                idx_swb.append(ISI[on_swb])

                spike_swb_start[on_swb] = True
                spike_swb_all[on_swb] = True
                off_swb = on_swb
                while ISI[off_swb] < 0.16 and off_swb < taille_max:
                    off_swb += 1
                    idx_swb.append(ISI[off_swb])
                    spike_swb_all[off_swb] = True
                spike_swb_stop[off_swb] = True

                idx_swb_stop = np.hstack((idx_swb_stop, off_swb))
                nb_spike_in_burst = np.hstack((nb_spike_in_burst, (off_swb + 1) - on_swb))
                on_swb = off_swb + 1
                val_spike_in_burst.append(idx_swb)

            on_swb += 1

        self._spike_swb_percent = 100 * (nb_spike_in_burst.sum() / self._spike_times_rmz.size)

        all_info_swb = {'idx_swb_start': idx_swb_start, 'idx_swb_stop': idx_swb_stop,
                        'spike_swb_start': spike_swb_start, 'spike_swb_stop': spike_swb_stop,
                        'nb_spike_in_burst': nb_spike_in_burst, 'spike_swb_all': spike_swb_all,
                        'val_spike_in_burst': val_spike_in_burst}
        self.all_info_swb = all_info_swb
        # TODO extraire correctement les variables spike_swb

    @property
    def spike_times_swb(self):
        if self._spike_swb_percent is None:
            self._spikeinburst_()
        return self._spike_swb_percent

    def _spike_times_isi_(self) -> ndarray:
        if self._spike_times_isi.size <= 0:
            self.__interspikeinterval__()
        return self._spike_times_isi

    @property
    def spike_times_rmz(self) -> ndarray:
        if self._spike_times_rmz.size > 0:
            return self._spike_times_rmz

    def _spike_creation_(self):
        self.spike_times_isi = self._spike_times_isi_()
        self.start(self._spike_times_rmz)


class SpikeSegment(object):
    """
            Un segment correspond au découpage fait lors des manip : 
            - Lors de plusieurs sessions dans le meme enregistrement (Pharmaco) chaque session correspond à un segment
        la création des segments passe directement par les informations contenue dans les Events.
        Chaque début de segment est caractérisé par un start et se termine par un stop.
    """
    def __init__(self, raw_signal_file: str):
        self.neurones = None
        self._spike_raw: ndarray([float]) = np.array([], dtype=float)
        self._spike_rmz: ndarray([float]) = np.array([], dtype=float)
        self._spike_segment: Dict[str, ndarray] = {}
        self.basetime: Dict[str,ndarray] = {}
        self.time_start_stop: Dict[str, ndarray] = {}
        self.name: str = None
        self.path: Path = None
        self.raw_signal_file: str = raw_signal_file
        self._creat_base_time_()

    def _creat_base_time_(self):
        self.signal: ImportRawSignal = ImportRawSignal(self.raw_signal_file)
        _tmp_items: list = [i for i in self.signal.signal_segments.keys()]
        self.basetime.update([{items.split(',')[0]: v.base_time} for items,v in self.signal.signal_segments.items()][0])

    def __repr__(self):
        return 'SpikeSegment'

    def __load__(self):
        """
        Importation des temps des spikes brutes directement du fichier txt
        :return: 
        """
        print(f'path : {self.path}')
        with self.path.open() as file:
            for line in file:
                self._spike_raw = np.hstack((self._spike_raw, float(line)))
        self.__rmz__()

    def __rmz__(self):
        """
        global_t_start est la même valeur pour l'ensemble des neurones, mis en seconde
        permet de remettre a zero les valeurs natives données par neuralynx
        :return:
        """
        _tmp_items_segment = [i for i in self.signal.Event.keys()]
        temp = self._spike_raw / 10e5
        start: float = self.signal.Event[_tmp_items_segment[0]].event_signals.global_t_start
        self._spike_rmz = temp - start

    def _decoupe_spike_par_segment_(self):
        """

        :return: 
        """
        tmp: list = [{i: v.event_signals.start_stop_segment} for i, v in self.signal.Event.items() if i.find('start') != -1]
        for idx in range(len(tmp)):
            self.time_start_stop.update(tmp[idx])
        for items, val in self.time_start_stop.items():
            start: float = float(val[0][0][0])
            stop: float = float(val[1][0][0])
            temp = [self._spike_rmz[(self._spike_rmz > start) & (self._spike_rmz < stop)]]
            self._spike_segment.update({items.split(',')[0]: temp})


    def makespike(self):
        """
        traitement global de la manip avec toutes les segments (session) par neurone.
        :return:
        """
        self.__load__()
        self._decoupe_spike_par_segment_()
        self.neurones = [SpikeFiles(self._spike_segment[i], self.basetime[[i for i in self._spike_segment.keys()][0]]) for i in self._spike_segment.keys()]

        for i in range(len(self.neurones)):
            self.neurones[i]._spike_times_rmz[0]
            self.neurones[i].spike_times_rmz
            self.neurones[i].spike_times_isi
            self.neurones[i].spike_times_swb
            self.neurones[i]._spike_creation_()


class NeuroneThread(Thread):

    def __init__(self, func, args):
        super().__init__()
        self.func = func
        self.args = args

    def run(self):
        print(f'Démarrage thread courant : {thread.currentThread()}')
        self.func(*self.args)


class LoadSpikeFiles(object):

    def __init__(self, Spike_file_folder: str, raw_signal_file: str):
        self.files_folder: List[str] = None
        self.spike_file_folder: str = Spike_file_folder
        self.raw_signal_file: str = raw_signal_file
        self._neurone: Dict[str, SpikeSegment] = {}

    def _creat_neurone_(self, idx):
        _tmp_neurone = SpikeSegment(self.raw_signal_file)
        _tmp_neurone.path = Path(self.files_folder[idx])
        _tmp_neurone.name = _tmp_neurone.path.name
        _tmp_neurone.makespike()
        self._neurone['neurone' + str(idx)] = _tmp_neurone

    def __load__(self):
        self.files_folder = glob.glob(self.spike_file_folder)
        print('Chargement')
        if not self.files_folder:
            print('dossier vide')
        else:
            print(self.files_folder)
            idx: int
            thread_list: list = []
            for idx, val in enumerate(self.files_folder):
                _tmp_thread = NeuroneThread(func=self._creat_neurone_, args=[idx])
                thread_list.append(_tmp_thread)
                _tmp_thread.start()
            for thread in thread_list:
                thread.join()

    def neurone(self):
        self.__load__()
        return self._neurone

if __name__ == "__main__":
    csc_dir: str = r'/data/cplx07 + bsl'
    ext: str = '\*.txt'
    spike_file_folder: str = r'D:\Dropbox\python\import_neuralynxv2\data\cplx07 + bsl\clustering' + ext

    spike_files = LoadSpikeFiles(spike_file_folder, csc_dir)
    t1 = chrono.time()
    neurones = spike_files.neurone()
    t2 = chrono.time()
    print(f'temps global : {t2 - t1}')