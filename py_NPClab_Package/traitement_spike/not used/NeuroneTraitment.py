import numpy as np
from numpy.core.multiarray import ndarray
from pathlib import Path
import glob
from typing import List, Dict
from load.old.Load_neuralynx import RawSignals, RawSignal
import time as chrono


class ImportRawSignal(RawSignals):
    def __init__(self, csc_dir: Path):
        super().__init__(csc_dir)
        self.load()


class SpikeSegment(object):
    def __init__(self, raw_signal_file: Path):
        self.neurones = None
        self._spike_raw: ndarray([float]) = np.array([], dtype=float)
        self._spike_rmz: ndarray([float]) = np.array([], dtype=float)
        self._spike_segment: List = []
        self._raw_segment: List = []
        self._event_segment: List = []
        self._event_segment_on: List = []

        self.signal = ImportRawSignal(raw_signal_file)
        self.name: str = None
        self.path: Path = None

    def __load__(self):
        print(f'path : {self.path}')
        with self.path.open() as file:
            for line in file:
                self._spike_raw = np.hstack((self._spike_raw, float(line)))

    def _decoupe_segment(self):
        i: str
        for i in self.signal.Event.keys():
            if i.find('start') != -1:
                print(i)
                start: float = float(self.signal.Event[i].event_signals.start_stop_segment[0][0])
                stop: float = float(self.signal.Event[i].event_signals.start_stop_segment[1][0])
                temp = [self._spike_rmz[(self._spike_rmz > start) & (self._spike_rmz < stop)]]
                self._spike_segment.append(temp)

    def _decoupe_signal_segment(self):
        i: str
        for i in self.signal.signal_segments.keys():
            items = [str]
            for idx in self.signal.signal_segments[i].voltage_signals.keys():
                items.append(idx)
            self._raw_segment.append(self.signal.signal_segments[i].voltage_signals[items[1]])

    def _decoupe_event_segment(self):
        i: str
        for i in self.signal.Event.keys():
            if i.find('start') != -1:
                self._event_segment.append(self.signal.Event[i].event_signals.event_each_times)
            else:
                self._event_segment_on.append(self.signal.Event[i].event_signals.event_each_times)

    def __rmz(self):
        """
        global_t_start est la même valeur pour l'ensemble des neurones
        :return:
        """
        temp = self._spike_raw / 10e5
        start: float = self.signal.Event['num segment : 0, num event start: 0'].event_signals.global_t_start
        self._spike_rmz = temp - start

    def makespike(self):
        self.__load__()
        self.__rmz()
        self._decoupe_segment()
        self._decoupe_event_segment()
        self._decoupe_signal_segment()
        self.neurones = [
            SpikeFiles(self._spike_segment[i], self._event_segment[i], self._event_segment_on[i], self._raw_segment[i])
            for i in range(len(self._spike_segment))]
        for i in range(len(self.neurones)):
            self.neurones[i]._spike_times_rmz[0]
            self.neurones[i].spike_times_rmz
            self.neurones[i].spike_times_isi
            self.neurones[i].spike_times_swb
            self.neurones[i].spike_waveform()


class ExtractEvent(object):

    def __decoupe_event(self, event_time, signal):

        s1 = chrono.time()
        longueur = signal.time.size
        bin = list(range(0, longueur, 512000))
        base_index = list(range(0, longueur))
        if bin[-1] < longueur:
            bin.append(longueur - bin[-1] + bin[-1])
        vector_time = []
        vector_event = []
        time = signal.time
        for i in range(len(bin) - 1):
            vector_time.append([base_index[bin[i]:bin[i + 1]], time[bin[i]:bin[i + 1]]])
            vector_event.append(
                event_time[(event_time > time[bin[i]]) & (event_time < time[bin[i + 1] - 1])])
        s2 = chrono.time()
        print(s2 - s1)
        return vector_time, vector_event

    def _extraction_event_times(self, event_time, signal) -> ndarray:

        event_time_in_raw: ndarray([int]) = np.array([], dtype=int)

        vector_event_time: ndarray([float])
        vector_time, vector_event_time = self.__decoupe_event(event_time, signal)
        print(f'Début de l"extraction des times des events')
        print(f' nombre d"event" : {len(event_time)}')

        s1 = chrono.time()
        for n in range(len(vector_time)):
            if vector_event_time[n].size == 0:
                pass
            else:
                for i in range(len(vector_event_time[n])):
                    posi_temp = np.where((vector_time[n][1] > float(vector_event_time[n][i]) - 1 / 32000) & (
                            vector_time[n][1] < float(vector_event_time[n][i]) + 1 / 32000))
                    event_time_in_raw = np.hstack((event_time_in_raw, vector_time[n][0][posi_temp[0][0]]))
        s2 = chrono.time()
        print(s2 - s1)
        print(f'Fin de l"extraction des times des spikes')
        return event_time_in_raw


class ExtractSpike(object):

    def __decoupe_time(self, spike_time, signal):
        s1 = chrono.time()
        longueur = signal.time.size
        bin = list(range(0, longueur, 512000))
        base_index = list(range(0, longueur))
        if bin[-1] < longueur:
            bin.append(longueur - bin[-1] + bin[-1])
        vector_time = []
        vector_spike_time = []
        time = signal.time
        for i in range(len(bin) - 1):
            vector_time.append([base_index[bin[i]:bin[i + 1]], time[bin[i]:bin[i + 1]]])
            vector_spike_time.append(
                spike_time[(spike_time > time[bin[i]]) & (spike_time < time[bin[i + 1] - 1])])
        s2 = chrono.time()
        print(s2 - s1)
        return vector_time, vector_spike_time

    def _extraction_times(self, spike_time, signal):
        spike_time_in_raw: ndarray([int]) = np.array([], dtype=int)
        vector_time, vector_spike_time = self.__decoupe_time(spike_time, signal)
        print(f'Début de l"extraction des times des spikes')
        print(f' nombre de spike : {len(spike_time)}')
        s1 = chrono.time()
        for n in range(len(vector_time)):
            for i in range(len(vector_spike_time[n])):
                posi_temp = np.where((vector_time[n][1] > vector_spike_time[n][i] - 1 / 32000) & (
                        vector_time[n][1] < vector_spike_time[n][i] + 1 / 32000))
                spike_time_in_raw = np.hstack((spike_time_in_raw, vector_time[n][0][posi_temp[0][0]]))
        s2 = chrono.time()
        print(s2 - s1)
        print(f'Fin de l"extraction des times des spikes')
        return spike_time_in_raw


class Extract(ExtractEvent, ExtractSpike):

    def __init__(self):
        super().__init__()
        self.signal = None
        self.spike_time_in_raw: ndarray([int]) = np.array([], dtype=int)
        self.event_time_in_raw: ndarray([int]) = np.array([], dtype=int)
        self.event_time_in_raw_on: ndarray([int]) = np.array([], dtype=int)
        self._spike_waveform: ndarray([float]) = np.empty([50, ], dtype=float)

    def start(self, event_time, event_time_on, spike_time):
        self.event_time_in_raw = self._extraction_event_times(event_time, self.signal)
        self.event_time_in_raw_on = self._extraction_event_times(event_time_on, self.signal)
        self.spike_time_in_raw = self._extraction_times(spike_time, self.signal)
        # TODO : identifier les besoins d'extraction des waveform


class SpikeFiles(Extract):

    def __init__(self, spike_times_rmz: ndarray, event_times_rmz: ndarray, event_times_rmz_on: ndarray,
                 signal: RawSignal):
        super().__init__()
        self.signal: RawSignal = signal
        self._event_time_rmz: ndarray([float]) = event_times_rmz
        self._event_time_rmz_on: ndarray([float]) = event_times_rmz_on

        self._spike_times_rmz: ndarray([float]) = spike_times_rmz[0]
        self._spike_times_isi: ndarray([float]) = np.array([], dtype=float)
        self._spike_swb_percent: float = None
        self.all_info_swb: Dict[str, ndarray] = {}

    def __interspikeinterval(self):
        self._spike_times_isi = self._spike_times_rmz[1:] - self._spike_times_rmz[:-1]

    def _spikeinburst(self):
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
            self._spikeinburst()
        return self._spike_swb_percent

    @property
    def spike_times_isi(self) -> ndarray:
        if self._spike_times_isi.size <= 0:
            self.__interspikeinterval()
        return self._spike_times_isi

    @property
    def spike_times_rmz(self) -> ndarray:
        if self._spike_times_rmz.size > 0:
            return self._spike_times_rmz

    def spike_waveform(self):
        self.start(self._event_time_rmz, self._event_time_rmz_on, self.spike_times_rmz)


class LoadSpikeFiles(object):

    def __init__(self, Spike_file_folder: Path, raw_signal_file: Path):
        self.files_folder: List[str] = None
        self.spike_file_folder: Path = Spike_file_folder
        self.raw_signal_file: Path = raw_signal_file
        self.raw_signal: RawSignals = None
        self._neurone: Dict[str, SpikeSegment] = {}

    def __load__(self):
        self.files_folder = glob.glob(self.spike_file_folder)
        if self.files_folder is None:
            print('dossier vide')
        else:
            print(self.files_folder)
            for idx, val in enumerate(self.files_folder):
                self._neurone['neurone' + str(idx)] = SpikeSegment(self.raw_signal_file)
                self._neurone['neurone' + str(idx)].path = Path(self.files_folder[idx])
                self._neurone['neurone' + str(idx)].name = self._neurone['neurone' + str(idx)].path.name
                print(f'Traitement du neurone : {val}')
                self._neurone['neurone' + str(idx)].makespike()

        return self._neurone

    @property
    def neurone(self):
        if self._neurone.__len__() == 0:
            self.__load__()
        return self._neurone
