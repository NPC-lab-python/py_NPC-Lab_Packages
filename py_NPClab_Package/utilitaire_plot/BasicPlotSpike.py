# -*-coding:utf-8 -*-
# utilitaire_plot package -->

import numpy as np
import matplotlib.pyplot as plt

from py_NPClab_Package.utlilitaire_saving.Saving_traitment import SavingMethodes
from sklearn.neighbors import KernelDensity
from numpy.core.multiarray import ndarray
from typing import List
from py_NPClab_Package.utilitaire_load.basic_load import LoadData, ImportNeuralynx
from py_NPClab_Package.traitement_event.EventTraitement import EventFileNeuralynx
import pandas as pd
from pandas import DataFrame
from pandas import Series
from py_NPClab_Package.utilitaire_traitement.Decorateur import mesure
from py_NPClab_Package.utilitaire_neuralynx.Load_neuralynx import GlobalRawSignal

import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


class PlotSpike(object):

    def __init__(self):
        pass

    @mesure
    def _calcul_swb_glissant_(self, spike_times_rmz: ndarray, spike_times_isi: ndarray, taille_fenetre: int, pas_de_gliss: int):

        nb_bin = [np.arange(spike_times_rmz[0], spike_times_rmz[-1] - taille_fenetre, pas_de_gliss),
                  np.arange(spike_times_rmz[0] + taille_fenetre, spike_times_rmz[-1], pas_de_gliss)]

        nb_swb = np.array([], dtype=list)

        for i in range(nb_bin[0].size):
            spike_times_rmz2 = np.where((spike_times_rmz > nb_bin[0][i]) & (spike_times_rmz < nb_bin[1][i]))
            on_swb: int = spike_times_rmz2[0].min()
            off_swb: int = None
            taille_max: int = spike_times_rmz2[0].max()
            ISI = spike_times_isi
            nb_spike_in_burst = np.array([], dtype=int)
            while on_swb < taille_max:
                if ISI[on_swb] < 0.08:
                    off_swb = on_swb
                    while ISI[off_swb] < 0.16 and off_swb < taille_max:
                        off_swb += 1
                    nb_spike_in_burst = np.hstack((nb_spike_in_burst, (off_swb + 1) - on_swb))
                    on_swb = off_swb + 1
                on_swb += 1
            spike_percent = 100 * (nb_spike_in_burst.sum() / spike_times_rmz2[0].size)
            nb_swb = np.hstack((nb_swb, spike_percent))
        return nb_swb, nb_bin

    def _calcul_frequence_glissant_(self, spike_times_rmz: ndarray, taille_fenetre: int, pas_de_gliss: int):
        """

        :param spike_times_rmz: temps des PA en sec
        :param taille_fenetre: en sec ex 15
        :param pas_de_gliss: en sec ex 5
        :return:
        """

        nb_bin = [np.arange(spike_times_rmz[0], spike_times_rmz[-1] - taille_fenetre, pas_de_gliss),
                  np.arange(spike_times_rmz[0] + taille_fenetre, spike_times_rmz[-1], pas_de_gliss)]
        freq_mean = np.array([], dtype=float)

        for i in range(nb_bin[0].size):
            temp_spike = spike_times_rmz[(spike_times_rmz >= nb_bin[0][i]) & (spike_times_rmz < nb_bin[1][i])]
            freq_mean = np.hstack((freq_mean, (1 / taille_fenetre) * temp_spike.size))
        return freq_mean, nb_bin

    def _correlogram_(self, *args):
        """
        lag max correspond à la fenetre dans laquel on regarde en sec
        :param args:
        :return:
        """
        lag_max: int = args[0]
        neuroneA: ndarray([float]) = args[1]
        if len(args) == 4:
            neuroneB: ndarray([float]) = neuroneA
            range_bin: float = args[2]
            nb_bin: int = args[3]
        elif len(args) == 5:
            neuroneB: ndarray([float]) = args[2]
            range_bin: float = args[3]
            nb_bin: int = args[4]
        else:
            print('Problèmes')

        temp: ndarray([float])
        lag: ndarray([float]) = np.array([], dtype=float)
        for i, val in enumerate(neuroneA):
            temp = neuroneA[i] - neuroneB[(neuroneB > neuroneA[i] - lag_max) & (neuroneB < neuroneA[i] + lag_max) &
                                          (neuroneB != neuroneA[i])]
            lag = np.hstack((lag, temp))
        hist, bin_edges = np.histogram(lag, bins=nb_bin, range=(-range_bin, range_bin), density=True)
        return lag, hist, bin_edges


class GenericPlotV2(PlotSpike, SavingMethodes):
    _thickness: float = 1
    _number_of_point: int = 25

    def __init__(self):
        super().__init__()
        # self._number_of_point = 25
        # self._thickness = 1

    @classmethod
    def __set_thickness__(cls, val: float):
        cls._thickness = val

    @classmethod
    def __set_number_of_point__(cls, val: int):
        cls._number_of_point = val


    def _axis_list_(self, items: List[str], fig1):
        axis_list = []
        if len(items) != 1:
            for ind_axsubplot in range(1, len(items) + 1):
                axis_list.append(fig1.add_subplot(np.ceil(len(items) / 2), 2, ind_axsubplot))
        else:
            axis_list.append(fig1.add_subplot(len(items), 1, 1))
        return axis_list

    def _crea_fig_(self):
        fig = plt.figure(4, figsize=(20, 20))
        return fig

    def _insertspikeinplotraw_(self, ax, base_time: ndarray, items_voltage_signal, ind_subplot, spike_index_in_raw, raw_signal: GlobalRawSignal,
                               colors):
        """

        :param ax:
        :param base_time:
        :param items_voltage_signal:
        :param ind_subplot:
        :param spike_time_in_raw:
        :param voltage_signals:
        :param colors:
        :return:
        """
        for i in spike_index_in_raw:
            ax.plot(base_time[i - self._number_of_point:i + self._number_of_point],
                    raw_signal.voltage_signals[items_voltage_signal[ind_subplot]].magnitude[i - self._number_of_point:i + self._number_of_point],
                    colors,
                    linewidth=1)
        return ax

    def _inserteventinplotraw_(self, ax, base_time: ndarray, event_index_in_raw, event_trace):
        thickness: float = 1
        for i in event_index_in_raw:
            ax.plot([base_time[int(i)], base_time[int(i)]], [event_trace[int(i)] - 200,
                                                             event_trace[int(i)]], color='blue', linewidth=thickness)
        return ax

    def plot_waveform_neurone(self, neurones_index_val: List[dict], raw_signal: GlobalRawSignal, event: EventFileNeuralynx,
                                intervalle: List[int] = None, name: str = None, option_csc: List[str] = None):

        colors = ['red', 'b', 'g', 'y', 'black']

        fig1 = self._crea_fig_()
        items_voltage_signal = [i for i in raw_signal.voltage_signals.keys() if i in option_csc]

        axis_list = self._axis_list_(items_voltage_signal, fig1)

        x_spike = np.arange(0, self._number_of_point*2)
        name_neurone: str
        for idx, neurone in enumerate(neurones_index_val):
            spike_index_in_raw = neurone[(neurone > intervalle[0]) & (neurone <intervalle[1])]


            for ind_subplot, ax in enumerate(axis_list):
                for i in spike_index_in_raw:
                    ax.plot(x_spike, raw_signal.voltage_signals[items_voltage_signal[ind_subplot]].magnitude[
                                     i - self._number_of_point:i + self._number_of_point],
                            colors[idx], linewidth=self._thickness)
                ax.set_xlabel(raw_signal.voltage_signals[items_voltage_signal[ind_subplot]].info_channels)
                ax.set_ylabel('µV')

        self._save_figure_(fig=fig1, name=name, option='waveform')

    def plot_neurone_in_raw(self, neurones_index_val: List[dict], raw_signal: GlobalRawSignal,
                            event: DataFrame, option_other_trigger: Series = pd.Series([]),
                            intervalle: List[int] = None, name: str = None,
                            option_csc: List[str] = None):
        """
                Permet de plot un ou plusieurs neurones_index_val sur la/les trace brute choisi dans option_csc.
                Si aucun csc est choisi (option_csc), toutes les traces contenues dans le dossier seront plotées
                Si aucun neurone est choisi (choix_neurone), ils seront tous plotés


        :param intervalle: il doit correspondre a un index de callage sur "raw_signal.base_time"
        :param neurone:
        :param segment:
        :param choix_neurone:
        :param name:
        :param option_csc:
        :return:
        """
        if not intervalle:
            intervalle: List[int] = [raw_signal.base_time.min(), raw_signal.base_time.max()]

        colors = ['red', 'b', 'black', 'g', 'red', 'b', 'black', 'g']

        fig1 = self._crea_fig_()
        items_voltage_signal = [i for i in raw_signal.voltage_signals.keys() if i in option_csc]

        axis_list = self._axis_list_(items_voltage_signal, fig1)

        event_trace: ndarray([int]) = np.ones([raw_signal.base_time.size], dtype=int) * 300

        for idx, neurone in enumerate(neurones_index_val):
            base_time: ndarray = raw_signal.base_time
            spike_index_in_raw = neurone[(neurone > intervalle[0]) & (neurone <intervalle[1])]
            # event_index_in_raw = event['num segment : 1']['index'][(event['num segment : 1']['index'] > intervalle[0]) & (event['num segment : 1']['index'] <intervalle[1])]

            if option_other_trigger.size <1:
                event_index_in_raw = event['index'][
                    (event['index'] > intervalle[0]) & (
                                event['index'] < intervalle[1])]
            else:
                event_index_in_raw = event['index'][
                    (event['index'] > intervalle[0]) & (
                                event['index'] < intervalle[1])]
                other_trigger_index_in_raw = option_other_trigger[(option_other_trigger > intervalle[0]) & (option_other_trigger <intervalle[1])]

            for ind_subplot, ax in enumerate(axis_list):
                ax.plot(base_time[intervalle[0]:intervalle[1]],
                        raw_signal.voltage_signals[items_voltage_signal[ind_subplot]].magnitude[intervalle[0]:intervalle[1]],
                        'black', linewidth=0.1)

                ax = self._insertspikeinplotraw_(ax, base_time, items_voltage_signal, ind_subplot, spike_index_in_raw,
                                                 raw_signal, colors[idx])
                if option_other_trigger.size < 1:
                    ax = self._inserteventinplotraw_(ax, base_time, event_index_in_raw, event_trace)
                else:
                    ax = self._inserteventinplotraw_(ax, base_time, event_index_in_raw, event_trace)
                    ax = self._inserteventinplotraw_(ax, base_time, other_trigger_index_in_raw, event_trace)

                ax.set_xlabel(raw_signal.voltage_signals[items_voltage_signal[ind_subplot]].info_channels)
                ax.set_ylabel('µV')

        self._save_figure_(fig=fig1, name=name, option='allspike')

    def plot_raster_event_spike(self, spike_time: ndarray, time_event: Series, name: str):
        """
        normalisation entre -1 et 1 ! faire attention car si la fenetre est dif de 1 la base temporelle restara entre -1 et 1
        (2*(e.loc[3][e.loc[3].notna()]-(time_event[3] - fenetre))/((time_event[3] + fenetre)-(time_event[3] - fenetre)))-1
        """

        fenetre = 2

        fig1 = self._crea_fig_()

        axis_list = []
        axis_list.append(fig1.add_subplot(1, 1, 1))

        # lineoffsets1 = np.array([1])
        # linelengths1 = [1.5]
        lineoffsets2 = 1
        linelengths2 = 1

        # colors1 = 'black'
        colors2 = 'r'
        e = pd.DataFrame()
        tmp = pd.DataFrame()
        tmp_serie = pd.Series(dtype=float)
        for id, val in enumerate(time_event):
            e = e.append(pd.Series(
                spike_time[((spike_time < time_event[id] + fenetre) & (spike_time > time_event[id]-0.1))], name=str(id)))

        val: str
        for ind_subplot, ax in enumerate(axis_list):
            for idx, val in enumerate(time_event):
                # _spike_around_norma = (2*(e.iloc[idx][e.iloc[idx].notna()]-(time_event[idx] - fenetre))/((time_event[idx] + fenetre)-(time_event[idx] - fenetre)))-1
                _spike_around_norma = e.iloc[idx][e.iloc[idx].notna()]-(time_event[idx])
                tmp_serie = tmp_serie.append(_spike_around_norma)

                tmp[str(idx)] = [_spike_around_norma]
                ax.eventplot(_spike_around_norma, linelengths=linelengths2, lineoffsets=lineoffsets2+idx, colors=colors2)
                # ax.eventplot(e.loc[2][e.loc[2].notna()],linelengths=linelengths2, lineoffsets=lineoffsets2, colors=colors2)
            ax.plot([0, 0], [0, idx],'o-')

        self._save_figure_(fig=fig1, name=name, option='raster')
        return tmp_serie, tmp

    def plot_kernel_density(self, _tmp, name: str):
        fig1 = self._crea_fig_()

        Vecvalues = _tmp[:, None]
        Vecpoints = np.linspace(-0.1, 2, 100)[:, None]
        kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(Vecvalues)
        logkde = kde.score_samples(Vecpoints)
        plt.plot(Vecpoints, np.exp(logkde))
        # fig1.show()

        self._save_figure_(fig=fig1, name=name, option='kdensity')

    def plot_frequence_glissante(self, neurones: List[ndarray], taille_fenetre: int, pas_de_gliss: int, name_neurone: List[str], name: str):
        """
        Les neurones sont directement importer.
        il faut que les temps des spikes soit en secondes dans un "np.array".
        ex:
            objet.plot_frequence_glissante(neurones=[np.array(neurone0.data['time']), np.array(neurone1.data['time'])],
                                  name_neurone=['neurone0', 'neurone1'],
                                  taille_fenetre=15, pas_de_gliss=5, name='neuron')
        :param neurones: liste des neurones importer [neurone.data['time']]
        :param taille_fenetre: 15 sec
        :param pas_de_gliss: 5 sec
        :param name: name du fichier de sortie
        :param name_neurone: liste des name de chaque neurone ex: ['neurone0', 'neurone1']
        :return:
        """
        fig1 = self._crea_fig_()
        axis_list = self._axis_list_(name_neurone, fig1)

        for ind_subplot, ax in enumerate(axis_list):
            freq_all = neurones[ind_subplot].size / (neurones[ind_subplot][-1] - neurones[ind_subplot][0])
            freq, nb_bin = self._calcul_frequence_glissant_(
                neurones[ind_subplot], taille_fenetre, pas_de_gliss)
            ax.plot(nb_bin[0], freq)
            ax.set(ylim=(0, 12))
            ax.set_xlabel(f'{ind_subplot} fréq mean : {str(freq_all)}')
            ax.set_ylabel('Fréquence (Hz)')

        self._save_figure_(fig=fig1, name=name, option='frequence')

    def plot_burst_glissant(self, neurones_times: List[Series], neurones_isi: List[Series], taille_fenetre: int, pas_de_gliss: int, name_neurone: List[str], name: str):

        fig1 = self._crea_fig_()
        axis_list = self._axis_list_(name_neurone, fig1)

        for ind_subplot, ax in enumerate(axis_list):

            nb_swb, nb_bin = self._calcul_swb_glissant_(np.array(neurones_times[ind_subplot]), np.array(neurones_isi[ind_subplot]), taille_fenetre, pas_de_gliss)

            ax.plot(nb_bin[0], nb_swb)
            ax.set(ylim=(0, 100))
            ax.set_xlabel(f'{ind_subplot} swb mean : {str(sum(nb_swb/len(nb_swb)))}')
            ax.set_ylabel('SWB (%)')

        self._save_figure_(fig=fig1, name=name, option='swbmean')

    def plothist(self, val, bin, name_neurone, data):
        fig1 = self._crea_fig_()
        axis_list = self._axis_list_(name_neurone, fig1)
        for ind_subplot, ax in enumerate(axis_list):
            # ax.hist(bin[:-1], bin, weights=val)
            ax.hist(data[1], 20)
        fig1.show()

    # def plot_first_spike_in_swb_neurone_in_raw(self, intervalle: List[int], neurone: Dict[str, SpikeSegment],
    #                                            event: EventFileNeuralynx, segment: int,
    #                                            name: str = None, choix_neurone: List[str] = None,
    #                                            option_csc: List[str] = None):
    #
    #     colors = ['red', 'b', 'black', 'g', 'red', 'b', 'black', 'g']
    #
    #     fig1 = self._crea_fig_()
    #     items_neurone, items_segment, items_voltage_signal = self._items_(segment, choix_neurone, option_csc)
    #     axis_list = self._axis_list_(items_voltage_signal, fig1)
    #     voltage_signals: Dict[str, RawSignal] = self.neurone[items_neurone[0]].signal.signal_segments[
    #         items_segment[segment]].voltage_signals
    #
    #
    #     event_trace: ndarray([int]) = np.ones([neurone[items_neurone[0]].basetime[items_segment[0].split(',')[0]].size], dtype=int) * 300
    #
    #     event_time_in_raw, event_time_in_raw_on = self._event_timming_(event, items_segment, intervalle)
    #
    #
    #     name_neurone: str
    #     for idx, name_neurone in enumerate(items_neurone):
    #         base_time: ndarray = neurone[name_neurone].neurones[segment].basetime
    #
    #         # extraction des premiers spikes des burst comprisent  uniquement dans les events
    #         spike_time_in_r = neurone[name_neurone].neurones[segment].spike_time_in_raw[
    #             neurone[name_neurone].neurones[segment].all_info_swb['idx_swb_start']]
    #
    #         spike_time_in_raw = spike_time_in_r[((spike_time_in_r > intervalle[0]) & (spike_time_in_r < intervalle[1]))]
    #
    #         for ind_subplot, ax in enumerate(axis_list):
    #             ax.plot(base_time[intervalle[0]:intervalle[1]],
    #                     voltage_signals[items_voltage_signal[ind_subplot]].magnitude[intervalle[0]:intervalle[1]],
    #                     'black', linewidth=0.1)
    #
    #             ax = self._insertspikeinplotraw_(ax, base_time, items_voltage_signal, ind_subplot, spike_time_in_raw,
    #                                              voltage_signals, colors[idx])
    #
    #             ax = self._inserteventinplotraw_(ax, base_time, items_voltage_signal, ind_subplot, event_time_in_raw,
    #                                              voltage_signals, event_trace)
    #             ax = self._inserteventinplotraw_(ax, base_time, items_voltage_signal, ind_subplot, event_time_in_raw_on,
    #                                              voltage_signals, event_trace)
    #             ax.set_xlabel(voltage_signals[items_voltage_signal[ind_subplot]].info_channels)
    #             ax.set_ylabel('µV')
    #
    #     self._save_figure_(fig=fig1, name=name, option='firstspike')
    @mesure
    def plotcorrelogram(self, neurones: List[Series], lag_max: float, lenght_of_bin: float, name: str):
        """

        :param neurones:
        :param lag_max: en sec
        :param lenght_of_bin: en sec
        :param name:
        :return:
        """

        range_bin = lag_max
        nb_bin = int(lag_max / lenght_of_bin)

        fig1, ax = plt.subplots(len(neurones), len(neurones))
        fig1.set_size_inches(20, 20)

        for idx in range(len(neurones)):
            if len(neurones) == 1:
                lag, hist, bin_edges = self._correlogram_(lag_max, neurones[idx], neurones[idx],
                                                          range_bin, nb_bin)
                ax.hist(bin_edges[:-1], bin_edges, weights=hist)
            else:
                for idy in range(len(neurones)):
                    if idx == idy:
                        ax[idx][idy].set_fc('black')
                        lag, hist, bin_edges = self._correlogram_(lag_max, neurones[idx], neurones[idy], range_bin, nb_bin)
                        ax[idx][idy].hist(bin_edges[:-1], bin_edges, weights=hist)
                    else:
                        ax[idx][idy].set_fc('red')
                        lag, hist, bin_edges = self._correlogram_(lag_max, neurones[idx], neurones[idy], range_bin, nb_bin)
                        ax[idx][idy].hist(bin_edges[:-1], bin_edges, weights=hist)
                        ax[idx][idy].set_xlabel('neurone' + str(idx) + ' : ' + 'neurone' + str(idy))
                        # ax[idx][idy].set_ylabel('Fréquence (Hz)')

        self._save_figure_(fig=fig1, name=name, option='crosscorrelogramme')

if __name__ == "__main__":
    # -----------------------------------------------partie raw signal -------------------------------------
    dir_data: str = r'/data/cplx07 + bsl'
    #
    raw_brute = LoadData.init_data(ImportNeuralynx, dir_data, 'csc')
    # # -------------------------------------------- partie event ---------------------------------------------------
    # from utilitaire_load.basic_load import ImportNeuralynx, LoadData
    # from traitement_event.EventTraitement import EventFileNeuralynx, EventRewardNeuralynx
    #
    dir_data: str = r'/data/cplx07 + bsl'
    #
    event_brute = LoadData.init_data(ImportNeuralynx, dir_data, 'event')

    Event = EventFileNeuralynx(dir_data=dir_data, event=event_brute)
    info_event, all_event = Event.set_event()
    #
    # reward = EventRewardNeuralynx()
    #
    # start_stim, start_stim_index = reward.set_reward(reward_time=all_event['num segment : 1']['time'],
    #                                                  reward_index=all_event['num segment : 1']['index'])
    # start_stim_index = start_stim_index.astype(dtype=int)



    #
    # # ------------------------------------ parti spike -----------------------------------------
    #
    # from traitement_spike.NeuroneTraitment import Segment, Spike, PreFormatSpike
    # from utilitaire_load.basic_load import NeuralynxFilesSpike
    #
    # dir_spikefile: str = r'D:\Dropbox\python\import_neuralynxv2\data\cplx07 + bsl\clustering\*.txt'
    #
    # spikefiles = LoadData.init_data(NeuralynxFilesSpike, dir_spikefile)
    #
    #
    #
    # segment_infos = Segment(dir_data=dir_data)
    # segment_data = segment_infos.set_segment(event_brute=event_brute.Event, event_final=all_event)
    #
    # e = PreFormatSpike(dir_data=dir_data, spike_files=spikefiles, segment_infos=segment_data, all_event=all_event)
    # e.set_spike()
    #
    # spike = Spike(dir_data=dir_data, name_neurone='neurone0_brute', segment_infos=segment_data, num_segment=0)
    # spike.set_neurone()
    # spike = Spike(dir_data=dir_data, name_neurone='neurone0_brute', segment_infos=segment_data, num_segment=1)
    # spike.set_neurone()
    # spike = Spike(dir_data=dir_data, name_neurone='neurone1_brute', segment_infos=segment_data, num_segment=0)
    # spike.set_neurone()
    # spike = Spike(dir_data=dir_data, name_neurone='neurone1_brute', segment_infos=segment_data, num_segment=1)
    # spike.set_neurone()
    #
    # del spike, segment_infos, segment_data, e


    # -------------------------------------------- partie reload specifique neurone ---------------------------------
    from py_NPClab_Package.utilitaire_load import NeuroneFilesSerialiser

    dir_save = r'/data/cplx07 + bsl/save'
    name = 'segment1_neurone0'
    neurone0 = LoadData.init_data(NeuroneFilesSerialiser, dir_save, name)
    name = 'segment1_neurone1'
    neurone1 = LoadData.init_data(NeuroneFilesSerialiser, dir_save, name)
    plot = GenericPlotV2()
    # plot.plot_frequence_glissante(neurones=[np.array(neurone0.data['time']), np.array(neurone1.data['time'])],
    #                               name_neurone=['neurone0', 'neurone1'],
    #                               taille_fenetre=15, pas_de_gliss=5, name='neuron')

    # plot.plot_burst_glissant(neurones=[neurone0.data, neurone1.data],
    #                               name_neurone=['neurone0', 'neurone1'],
    #                               taille_fenetre=15, pas_de_gliss=5, name='neuron')

    # plot.plotcorrelogram(neurones=[neurone0.data, neurone1.data], lag_max=0.1, lenght_of_bin=0.0005, name='cross')

    # ------------------------------------- partie plot raw signal ----------------------------------------

    intervalle = [150 * 32000, 164 * 32000]  # intervalle par seconde

    plot.plot_neurone_in_raw(intervalle=intervalle, neurones_index_val=[neurone0.data['time_index_in raw'], neurone1.data['time_index_in raw']], raw_signal=raw_brute.signal_segments['num segment : 1'],
                             event=all_event,
                             name='raw plot', option_csc=['CSC2', 'CSC6', 'CSC7', 'CSC10'])

    plot.plot_waveform_neurone(intervalle=intervalle, neurones_index_val=[neurone0.data['time_index_in raw'], neurone1.data['time_index_in raw']], raw_signal=raw_brute.signal_segments['num segment : 1'],
                             event=all_event,
                             name='raw plot', option_csc=['CSC2', 'CSC6', 'CSC7', 'CSC10'])