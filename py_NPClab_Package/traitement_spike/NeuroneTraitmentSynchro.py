# -*-coding:utf-8 -*-

import time as chrono
from collections import defaultdict
from typing import List, Dict

import numpy as np
import pandas as pd
from numpy.core.multiarray import ndarray
from pandas import DataFrame, Series
from quantities.quantity import Quantity

from py_NPClab_Package.traitement_event.EventTraitement import EventFileNeuralynx, ExtractEvent, EventRewardNeuralynx
from py_NPClab_Package.utilitaire_neuralynx.Load_neuralynx import GlobalEventBasetime
from py_NPClab_Package.utilitaire_load.basic_load import LoadData, ImportNeuralynx, NeuralynxFilesSpike
from py_NPClab_Package.utlilitaire_saving.Saving_traitment import SaveSerialisation
from py_NPClab_Package.utilitaire_traitement.Decorateur import mesure

import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


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

    @mesure
    def set_segment(self, event_brute, event_final):
        saving = self._get_conf_(dir_save_conf=self.dir_data, name='segment_infos')
        if saving[0] == 1:
            self.segment_infos = defaultdict(GlobalEventBasetime)
            self.items_event = self._liste_event_(event_brute=event_brute)
            self.itmes_segment = self._liste_segment_(event_brute=event_brute)
            self.segment_infos = self._set_segment_infos_(event_brute, self.itmes_segment, event_final)
            self._set_conf_(name='segment_infos', dir_save_conf=saving[1], data=self.segment_infos)
            return self.segment_infos
        else:
            logging.debug(f'Chargement du fichier conf')
            self.segment_infos = self._chargement_fichier_data_(path=saving[1], name='segment_infos')
            return self.segment_infos


class PreFormatSpike(SaveSerialisation):
    """
    Cette class permet de préparer le fichier spike pour en suite etre traiter par la class "Spike"
    elle enregistre un fichier serialiser contenant les temps par segment.

    Pour utiliser cette class :


    """

    def __init__(self, dir_data: str, spike_files: NeuralynxFilesSpike, segment_infos: defaultdict, all_event: defaultdict):
        """


        :spike_files: correspond aux fichier ".txt" brutes sous la forme final d'un dataframe
        :segment_infos:
        :all_event:
        """
        self.dir_data: str = dir_data
        self.segment_infos: defaultdict = segment_infos
        self.spike_files: NeuralynxFilesSpike = spike_files
        self.all_event: defaultdict = all_event
        self.start_stop_spike_time = defaultdict(DataFrame)

    @mesure
    def set_spike(self):
        """
         Cette méthode permet la création des fichiers intermediaires "neuroneX_brute.dat"
         qui contient l'ensemble des temps des spikes brutes.

        """
        start_stop_spike_time = pd.DataFrame()

        item_segment = list(self.segment_infos.keys())
        item_global_t_start = list(self.segment_infos[item_segment[0]].keys())[1]
        global_t_start = self.segment_infos[item_segment[0]][item_global_t_start]
        i: tuple
        for i in self.spike_files.data.items():

            name_file_spike = list(self.spike_files.data[i[0]].keys())[0].split('.')[0]

            saving = self._get_conf_(dir_save_conf=self.dir_data, name=f'{name_file_spike}_brute')
            if saving[0] == 1:
                self.spike_files.data[i[0]]['rmz_global'] = self._rmz_(i[1], global_t_start)
                for item in item_segment:
                    logging.debug(f'Début traitement du fichier de spikefiles {name_file_spike} '
                                  f'- recallage des temps en fonction du '
                                  f'segment : {item}')

                    start_stop_spike_time[item] = self._start_stop_(self.all_event[item]['time'], item)
                    self.start_stop_spike_time = start_stop_spike_time
                    self.spike_files.data[i[0]][item] = self._decoupe_spike_par_segment_(
                        self.spike_files.data[i[0]]['rmz_global'], start_stop_spike_time[item])
                    self.spike_files.data[i[0]]['name_file'] = name_file_spike

                self._set_conf_(name=f'{name_file_spike}_brute', dir_save_conf=saving[1], data=self.spike_files.data[i[0]])
            else:
                logging.debug(f'Chargement du fichier de spike_brute {name_file_spike}')
                self.spike_files.data[i[0]] = self._chargement_fichier_data_(path=saving[1],
                                                                             name=f'{name_file_spike}_brute')


    def _rmz_(self, spike: DataFrame, global_t_start) -> Series:
        """
        global_t_start est la même valeur pour l'ensemble des neurones, mis en seconde
        permet de remettre a zero les valeurs natives données par neuralynx

        :spike: correspond au contenu du fichier "spike_files" donc temps brute
        :return:
        """
        temp = spike[spike.columns[0]] / 10e5
        spike_rmz = temp - global_t_start
        return spike_rmz

    def _start_stop_(self, start_stop: Series, item: str) -> Series:
        """
        Cette méthode récupère les temps des  events (all_event) correspondant
        directement au temps du segment correspondant.

        :start_stop:  all_event[item]['time']
        :item: correspond au nom du fichier du neurone
        """
        time_segment = pd.Series([start_stop.iloc[0], start_stop.iloc[-1]], name=item)
        return time_segment

    def _decoupe_spike_par_segment_(self, spike: Series, start_stop: Series) -> Series:
        """
        cette méthode permet de construire le spike_time correspondant au segment voulu

        :spike: correspond au temps des spikes recallé sur zéro obtenue avec "self._rmz_"
        :start_stop:
        :return:
        """
        time_spike = spike[(spike > start_stop.iloc[0]) & (spike < start_stop.iloc[1])]
        return time_spike


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
        self.basetime: ndarray = None
        self.spike_times_rmz: ndarray([float]) = None
        self.spike_times_isi: Series = None
        self.spike_time_in_raw: ndarray = None
        self.spike_swb_percent: float = None
        self.all_info_swb: Dict[str, ndarray] = {}

    def _set_base_time_(self) -> Quantity:
        item_segment = list(self.segment_infos.keys())
        segment_num = [i for i in item_segment if i.find(f'{self.num_segment}') != -1]
        basetime = self.segment_infos[segment_num[0]]['base_time']
        return basetime

    @mesure
    def set_neurone(self):
        logging.debug(f'Début du traitement des spikes pour le segment : {self.num_segment}'
                      f' du fichier brute : {self.name_neurone}')

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
            self.spike_time_in_raw: ndarray = self.basetime[self.spike_index_in_raw]
            data_save: dict = {'time_index_in raw': self.spike_index_in_raw, 'swb_percent': self.spike_swb_percent,
                         'time_brute': self.spike_time_in_raw, 'time': self.spike_times_rmz, 'isi': self.spike_times_isi}
            self._set_conf_(name=name_file, dir_save_conf=saving[1], data=data_save)

        else:
            logging.debug(f'chargement du fichier : {name_file}')
            self.spike_time_in_raw = self._chargement_fichier_data_(path=saving[1], name=name_file)

    def _pre_format_(self, event_time):
        event_time = pd.Series(Quantity(event_time, 's'))
        return event_time

    def __repr__(self):
        return 'SpikeFiles'

    def _interspikeinterval_(self) -> Series:
        spike_times_isi = self.spike_times_rmz.diff(1)
        return spike_times_isi

    @mesure
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


class CleaningSpikeTime(SaveSerialisation):

    def __init__(self, dir_data: str, name_neurone: str, start_stim: Series, time_ref_synchro: float):
        super().__init__()
        self.dir_data: str = dir_data
        self.start_stim: Series = start_stim
        self.name_neurone = name_neurone
        self.time_ref_synchro = time_ref_synchro


    def load_neurone(self) -> DataFrame:
        """
        cette methode permet de mettre en évidance les spike dans l'intervalle de stim
        Faire attention aux index car ils sont en float
        :return: une series booleen de la longueur de la series spike_time entrée
        """
        neurone = self.load_data_serializer(path=f'{self.dir_data}\\save', name=self.name_neurone)
        start = self.start_stim + self.time_ref_synchro
        ee = pd.DataFrame()
        r = pd.Series(False, index=pd.RangeIndex(0, len(neurone['time'])))
        e = pd.Series([], dtype=float)
        for i in start:
            e = e.append(neurone['time'][(neurone['time'] > i) & (neurone['time'] < i + 0.200)])
        r[e.index] = True
        ee['time_old'] = neurone['time']
        ee['time_cleaned_index'] = r
        ee['time_cleaned'] = e
        ee['time_deleted'] = pd.Series(np.array(ee['time_cleaned'].dropna()))
        ee['time_index_in raw'] = pd.Series(neurone['time_index_in raw'][~ee['time_cleaned_index']], dtype=int)
        ee['time'] = neurone['time'][~ee['time_cleaned_index']]
        ee['isi'] = ee['time'].dropna().diff()

        return ee

    def load_other(self) -> DataFrame:
        """
        cette methode permet de mettre en évidance les spike dans l'intervalle de stim
        Faire attention aux index car ils sont en float
        :return: une series booleen de la longueur de la series spike_time entrée
        """
        neurone = self.load_data_serializer(path=f'{self.dir_data}\\save', name=self.name_neurone)
        start = self.start_stim
        ee = pd.DataFrame()
        r = pd.Series(False, index=pd.RangeIndex(0, len(neurone['time'])))
        e = pd.Series([], dtype=float)
        for i in start:
            e = e.append(neurone['time'][(neurone['time'] > i) & (neurone['time'] < i + 0.200)])
        r[e.index] = True
        ee['time_old'] = neurone['time']
        ee['time_cleaned_index'] = r
        ee['time_cleaned'] = e
        ee['time_deleted'] = pd.Series(np.array(ee['time_cleaned'].dropna()))
        ee['time_index_in raw'] = pd.Series(neurone['time_index_in raw'][~ee['time_cleaned_index']], dtype=int)
        ee['time'] = neurone['time'][~ee['time_cleaned_index']]
        ee['isi'] = ee['time'].dropna().diff()
        return ee


if __name__ == "__main__":
    t1 = chrono.time()


    dir_spikefile: str = r'D:\Dropbox\python\import_neuralynxv2\data\cplx07 + bsl\clustering\*.txt'

    spikefiles = LoadData.init_data(NeuralynxFilesSpike, dir_spikefile)

    dir_data: str = r'/data/cplx07 + bsl'

    event_brute = LoadData.init_data(ImportNeuralynx, dir_data, 'event')

    Event = EventFileNeuralynx(dir_data=dir_data, event=event_brute)
    info_event, all_event = Event.set_event()

    reward = EventRewardNeuralynx()

    start_stim, start_stim_index = reward.set_reward(reward_time=all_event['num segment : 1']['time'],
                                                     reward_index=all_event['num segment : 1']['index'])
    start_stim_index = start_stim_index.astype(dtype=int)

    # ------------------------------ cleaning spike -----------------------------------
    new_spike = CleaningSpikeTime(dir_data= dir_data, name_neurone= 'segment1_neurone0', start_stim=start_stim)
    new_spike.load_neurone()

# -------------------------------

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