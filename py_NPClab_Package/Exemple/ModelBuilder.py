import numpy as np
from numpy.core.multiarray import ndarray
from typing import List, Dict
from quantities.quantity import Quantity
from NPClab_Package.utilitaire_load import LoadData, ImportNeuralynx, NeuralynxFilesSpike, GlobalEventBasetime
from NPClab_Package.traitement_event import EventFileNeuralynx, ExtractEvent
from collections import defaultdict
from utlilitaire_saving.Saving_traitment import SaveSerialisation
import pandas as pd
from pandas import DataFrame, Series
import time as chrono
import logging

logging.basicConfig(level=logging.DEBUG)


class Segment(SaveSerialisation):
    """
        Cette class permet la création d'un pseudo fichier de conf permettant le traiter des fichiers de neurones
        Pour utiliser cette class :


    """

    def __init__(self, dir_data: str):
        self.dir_data: str = dir_data

    def _liste_segment_(self, event_brute) -> List[str]:
        """
        retourne une liste unique des segmens présent
        :param event_brute: fichier direct de l'importation
        :return:
        """
        items_segment = list(set([i.split(',')[0] for i in event_brute.keys()]))
        return items_segment

    def _liste_event_(self, event_brute) -> List[str]:
        """
        retourne une liste unique des events présent
        :param event_brute: fichier direct de l'importation
        :return:
        """
        items_event = list(set([i.split(',')[1] for i in event_brute.keys() if i.find('num event') == 17]))
        return items_event

    def _set_segment_infos_(self, event_brute, items_segment, event_final):
        segment_infos = defaultdict(dict)
        for i in items_segment:
            segment_infos[i] = event_brute[i].__dict__
            segment_infos[i]['event_time'] = event_final[i]
        return segment_infos

    def set_segment(self, event_brute, event_final):
        t1 = chrono.time()

        saving = self._get_conf_(dir_save_conf=self.dir_data, name='segment_infos')
        if saving[0] == 1:
            self.segment_infos = defaultdict(GlobalEventBasetime)
            self.items_event = self._liste_event_(event_brute=event_brute)
            self.itmes_segment = self._liste_segment_(event_brute=event_brute)
            self.segment_infos = self._set_segment_infos_(event_brute, self.itmes_segment, event_final)
            self._set_conf_(name='segment_infos', dir_save_conf=saving[1], data=self.segment_infos)
            t2 = chrono.time()
            print(f'temps segment infos: {t2 - t1}')
            return self.segment_infos

        else:
            logging.debug(f'Chargement du fichier conf')
            self.segment_infos = self._chargement_fichier_data_(path=saving[1], name='segment_infos')
            t2 = chrono.time()
            print(f'temps segment  infos: {t2 - t1}')
            return self.segment_infos


class PreFormatSpike(SaveSerialisation):
    """
    Cette class permet de préparer le fichier spike pour en suite etre traiter par la class "Spike"
    Pour utiliser cette class :


    """

    def __init__(self, dir_data: str, spike_files, segment_infos, all_event):
        super().__init__()
        self.dir_data: str = dir_data
        self.segment_infos = segment_infos
        self.spike_files = spike_files
        self.all_event = all_event
        self.start_stop_spike_time = defaultdict(DataFrame)

    def set_spike(self):
        t1 = chrono.time()

        start_stop_spike_time = pd.DataFrame()

        item_segment = list(self.segment_infos.keys())
        item_global_t_start = list(self.segment_infos[item_segment[0]].keys())[1]
        global_t_start = self.segment_infos[item_segment[0]][item_global_t_start]
        i: tuple
        for i in self.spike_files.data.items():
            saving = self._get_conf_(dir_save_conf=self.dir_data, name=f'neurone{i[0]}_brute')
            if saving[0] == 1:
                self.spike_files.data[i[0]]['rmz_global'] = self._rmz_(i[1], global_t_start)
                for item in item_segment:
                    start_stop_spike_time[item] = self._start_stop_(self.all_event[item]['time'], item)
                    self.start_stop_spike_time = start_stop_spike_time
                    self.spike_files.data[i[0]][item] = self._decoupe_spike_par_segment_(
                        self.spike_files.data[i[0]]['rmz_global'], start_stop_spike_time[item])

                self._set_conf_(name=f'neurone{i[0]}_brute', dir_save_conf=saving[1], data=self.spike_files.data[i[0]])
            else:
                logging.debug(f'Chargement du fichier neurone{i[0]}')
                self.spike_files.data[i[0]] = self._chargement_fichier_data_(path=saving[1],
                                                                             name=f'neurone{i[0]}_brute')
        t2 = chrono.time()
        print(f'temps preformat spike: {t2 - t1}')

    def _rmz_(self, spike, global_t_start) -> Series:
        """
        global_t_start est la même valeur pour l'ensemble des neurones, mis en seconde
        permet de remettre a zero les valeurs natives données par neuralynx
        :return:
        """
        temp = spike[spike.columns[0]] / 10e5
        spike_rmz = temp - global_t_start
        return spike_rmz

    def _start_stop_(self, start_stop: Series, item):
        e = pd.Series([start_stop.iloc[0], start_stop.iloc[-1]], name=item)
        return e

    def _decoupe_spike_par_segment_(self, spike, start_stop: Series):
        """

        :return:
        """
        temp = spike[(spike > start_stop.iloc[0]) & (spike < start_stop.iloc[1])]
        return temp


class Spike(SaveSerialisation, ExtractEvent):
    """
    Pour utiliser cette class :
    - il faut d'abord génerer le fichier "segment_infos" qui ce trouvera dans le dossier "save"
    - il faut importer les fichiers txt contenant les temps des spikes brute via la class PreFormatSpike (voir)
    - indiquer le chemin principal
    _ indiquer le neurone que l'on veut traiter
    _ indiquer le segment que l'on veut traiter

    les spikes sont directement chargés du fichier sérialisé du neurone qui devrait ce trouver dans le dossier "save"
    """

    def __init__(self, dir_data: str, name_neurone: str, segment_infos, num_segment: int):
        super().__init__()

        self.dir_data: str = dir_data
        self.segment_infos = segment_infos
        self.item_segment = list(self.segment_infos.keys())
        self.name_segment: str = [i for i in self.item_segment if i.find(f'{num_segment}') != -1][0]
        self.spike_index_in_raw: ndarray = None
        self.name_neurone = name_neurone
        self.num_segment = num_segment
        self.basetime: Quantity = None
        self.spike_times_rmz: ndarray([float]) = None
        self.spike_times_isi: Series = None
        self.spike_time_in_raw: Quantity = None
        self.spike_swb_percent: float = None
        self.all_info_swb: Dict[str, ndarray] = {}

    def _set_base_time_(self) -> Quantity:
        item_segment = list(self.segment_infos.keys())
        segment_num = [i for i in item_segment if i.find(f'{self.num_segment}') != -1]
        basetime = self.segment_infos[segment_num[0]]['base_time']
        return basetime

    def set_neurone(self):
        t1 = chrono.time()
        name_file = 'segment' + self.name_segment[-1] + '_' + self.name_neurone[:-6]
        saving = self._get_conf_(dir_save_conf=self.dir_data, name=name_file)
        if saving[0] == 1:
            self.basetime = self._set_base_time_()

            spike_files = self._chargement_fichier_data_(path=saving[1], name=self.name_neurone)
            neurone = spike_files[self.name_segment]
            self.spike_times_rmz: ndarray([float]) = np.array(neurone[~neurone.isna()])
            self.spike_times_rmz: Series = self._pre_format_(self.spike_times_rmz)
            self.spike_times_isi: Series = self._interspikeinterval_()
            self._spikeinburst_()
            self.spike_index_in_raw: ndarray = self._extraction_event_times_(self.spike_times_rmz, self.basetime)
            self.spike_time_in_raw: Quantity = self.basetime[self.spike_index_in_raw]
            self._set_conf_(name=name_file, dir_save_conf=saving[1], data=self.spike_time_in_raw)
        else:
            logging.debug(f'chargement du fichier : {name_file}')
            self.spike_time_in_raw = self._chargement_fichier_data_(path=saving[1], name=name_file)

        t2 = chrono.time()
        print(f'temps spike: {t2 - t1}')

    def _pre_format_(self, event_time):
        event_time = pd.Series(Quantity(event_time, 's'))
        return event_time

    def __repr__(self):
        return 'SpikeFiles'

    def _interspikeinterval_(self) -> Series:
        spike_times_isi = self.spike_times_rmz.diff(1)
        return spike_times_isi

    def _spikeinburst_(self):
        on_swb: int = 0
        off_swb: int = 0
        taille_max: int = self.spike_times_isi[:-1].size

        ISI = self.spike_times_isi
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
                while ISI[off_swb] < 0.16 and off_swb < taille_max - 1:
                    off_swb += 1
                    idx_swb.append(ISI[off_swb])
                    spike_swb_all[off_swb] = True
                spike_swb_stop[off_swb] = True

                idx_swb_stop = np.hstack((idx_swb_stop, off_swb))
                nb_spike_in_burst = np.hstack((nb_spike_in_burst, (off_swb + 1) - on_swb))
                on_swb = off_swb + 1
                val_spike_in_burst.append(idx_swb)

            on_swb += 1

        self.spike_swb_percent = 100 * (nb_spike_in_burst.sum() / self.spike_times_rmz.size)

        all_info_swb = {'idx_swb_start': idx_swb_start, 'idx_swb_stop': idx_swb_stop,
                        'spike_swb_start': spike_swb_start, 'spike_swb_stop': spike_swb_stop,
                        'nb_spike_in_burst': nb_spike_in_burst, 'spike_swb_all': spike_swb_all,
                        'val_spike_in_burst': val_spike_in_burst}
        self.all_info_swb = all_info_swb
        # TODO extraire correctement les variables spike_swb


if __name__ == "__main__":
    t1 = chrono.time()

    dir_spikefile: str = r'D:\Dropbox\python\import_neuralynxv2\data\cplx07 + bsl\clustering\*.txt'

    spikefiles = LoadData.init_data(NeuralynxFilesSpike, dir_spikefile)

    dir_data: str = r'/data/cplx07 + bsl'

    event_brute = LoadData.init_data(ImportNeuralynx, dir_data, 'event')

    Event = EventFileNeuralynx(dir_data=dir_data, event=event_brute)
    info_event, all_event = Event.set_event()

    segment_infos = Segment(dir_data=dir_data)
    segment_data = segment_infos.set_segment(event_brute=event_brute.Event, event_final=all_event)

    e = PreFormatSpike(dir_data=dir_data, spike_files=spikefiles, segment_infos=segment_data, all_event=all_event)
    e.set_spike()

    spike = Spike(dir_data=dir_data, name_neurone='neurone0_brute', segment_infos=segment_data, num_segment=0)
    spike.set_neurone()
    spike = Spike(dir_data=dir_data, name_neurone='neurone0_brute', segment_infos=segment_data, num_segment=1)
    spike.set_neurone()
    spike = Spike(dir_data=dir_data, name_neurone='neurone1_brute', segment_infos=segment_data, num_segment=0)
    spike.set_neurone()
    spike = Spike(dir_data=dir_data, name_neurone='neurone1_brute', segment_infos=segment_data, num_segment=1)
    spike.set_neurone()

    del spike

    t2 = chrono.time()
    print(f'temps global : {t2 - t1}')