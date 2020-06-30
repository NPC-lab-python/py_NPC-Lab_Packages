import numpy as np
import matplotlib.pyplot as plt
from numpy.core.multiarray import ndarray
from typing import List, Dict
from load.old.Load_neuralynx_separate import RawSignal

from traitement_spike.old_version.NeuroneTraitment_threader_separate import LoadSpikeFiles, SpikeSegment
from NPClab_Package.traitement_event import EventFileNeuralynx
import time as chrono
from quantities.quantity import Quantity
from utlilitaire_saving.Saving_traitment import SavingMethodes

import pandas as pd
from sklearn.neighbors import KernelDensity


class PlotSpike(object):

    def __init__(self):
        pass


    def _calcul_swb_glissant_(self, spike_times_rmz, spike_times_isi, taille_fenetre: int, pas_de_gliss: int):

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

    def _calcul_frequence_glissant_(self, spike_times_rmz, taille_fenetre: int, pas_de_gliss: int):

        nb_bin = [np.arange(spike_times_rmz[0], spike_times_rmz[-1] - taille_fenetre, pas_de_gliss),
                  np.arange(spike_times_rmz[0] + taille_fenetre, spike_times_rmz[-1], pas_de_gliss)]
        freq_mean = np.array([], dtype=float)

        for i in range(nb_bin[0].size):
            temp_spike = spike_times_rmz[(spike_times_rmz >= nb_bin[0][i]) & (spike_times_rmz < nb_bin[1][i])]
            freq_mean = np.hstack((freq_mean, (1 / taille_fenetre) * temp_spike.size))
        return freq_mean, nb_bin

    def _correlogram_(self, *args):

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


class GenericPlot(PlotSpike, SavingMethodes):
    _thickness: float = 0.3
    _number_of_point: int = 25

    def __init__(self, neurone: Dict[str, SpikeSegment]):
        super().__init__()
        self.neurone: Dict[str, SpikeSegment] = neurone
        self.__verifneurone__()

    @classmethod
    def __set_thickness__(cls, val: float):
        cls._thickness = val

    @classmethod
    def __set_number_of_point__(cls, val: int):
        cls._number_of_point = val

    def _items_(self, segment, choix_neurone: List[str] = None, option_csc: List[str] = None):

        if choix_neurone is None:
            items_neurone: List[str] = [i for i in self.neurone.keys()]
        else:
            items_neurone = choix_neurone

        items_segment: List[str] = [i for i in self.neurone[items_neurone[0]].signal.signal_segments.keys()]

        if option_csc is None:
            items_voltage_signal: List[str] = [i for i in self.neurone[items_neurone[0]].signal.signal_segments[
                items_segment[segment]].voltage_signals.keys()]
        else:
            items_voltage_signal = option_csc

        return items_neurone, items_segment, items_voltage_signal

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

    def __verifneurone__(self):
        try:
            self.neurone['neurone0'].neurones[0]._spike_times_rmz.size == 0
        except:
            print('problèmes neurone')

    def _insertspikeinplotraw_(self, ax, base_time: ndarray, items_voltage_signal, ind_subplot, spike_time_in_raw, voltage_signals,
                               colors):

        for i in spike_time_in_raw:
            ax.plot(base_time[i - self._number_of_point:i + self._number_of_point],
                    voltage_signals[items_voltage_signal[ind_subplot]].magnitude[i - self._number_of_point:i + self._number_of_point], colors,
                    linewidth=self._thickness)
        return ax

    def _inserteventinplotraw_(self, ax, base_time: ndarray, event_time_in_raw, event_trace):
        thickness: float = 0.3
        for i in event_time_in_raw:
            ax.plot([base_time[i], base_time[i]], [event_trace[i] - 200, event_trace[i]], color='blue', linewidth=thickness)
        return ax

    def plot_event_spike_time(self, neurone: Dict[str, SpikeSegment], segment: int, deb: int, fin: int, name: str):

        items_neurone: List[str] = []
        for i in neurone.keys():
            items_neurone.append(i)

        items: List[str] = []
        for i in neurone[items_neurone[0]].signal.Event.keys():
            items.append(i)

        lineoffsets1 = np.array([1])
        linelengths1 = [1.5]
        lineoffsets2 = np.array([2])
        linelengths2 = [4]

        colors1 = 'black'
        colors2 = 'r'

        fig, ax = plt.subplots(len(neurone), 1)
        val: str
        idx: int
        for idx, val in enumerate(items_neurone):
            ax[idx].eventplot(neurone[val].neurones[segment].spike_times_rmz
                              [(neurone[val].neurones[segment].spike_times_rmz >
                                neurone[val].signal.Event[items[0]].event_signals.event_each_times[deb]) &
                               (neurone[val].neurones[segment].spike_times_rmz < neurone[val].signal.Event
                               [items[0]].event_signals.event_each_times[fin])],
                              colors=colors1,
                              lineoffsets=lineoffsets1,
                              linelengths=linelengths1)

            ax[idx].eventplot(neurone[val].signal.Event[items[0]].event_signals.event_each_times[deb:fin],
                              colors=colors2,
                              lineoffsets=lineoffsets2,
                              linelengths=linelengths2)

        fig.show()
        # names: str = name + 'event.eps'
        # fig.savefig(names, format='eps')

    def plot_event(self, event: EventFileNeuralynx, deb: float, fin: float):

        seuil = 0.05

        lineoffsets1 = np.array([1])
        linelengths1 = [1.5]
        lineoffsets2 = np.array([2.5])
        linelengths2 = [1]

        colors1 = 'black'
        colors2 = 'r'

        fig1 = self._crea_fig_()
        axis_list = self._axis_list_(['event'], fig1)

        for ind_subplot, ax in enumerate(axis_list):

            ax.eventplot(event.all, colors=colors1, linewidths=0.2, lineoffsets=lineoffsets1, linelengths=linelengths1)
            ax.eventplot(event.all[event.all.diff() > seuil], linewidths=0.2, colors=colors2, lineoffsets=lineoffsets2, linelengths=linelengths2)

            ax.set(xlim=(deb, fin), ylim=(0, 4))



        fig1.show()
        # names: str = name + 'event.eps'
        # fig.savefig(names, format='eps')

    def plotcorrelogram(self, lag_max: float, lenght_of_bin: float, segment: int, name: str):

        range_bin = lag_max
        nb_bin = int(lag_max / lenght_of_bin)
        neuroneA: Dict[str, SpikeSegment] = self.neurone
        neuroneB: Dict[str, SpikeSegment] = self.neurone

        fig1, ax = plt.subplots(len(neuroneA), len(neuroneB))
        fig1.set_size_inches(20, 20)

        for idx in range(len(neuroneA)):
            for idy in range(len(neuroneB)):
                if idx == idy:
                    ax[idx][idy].set_fc('black')
                    lag, hist, bin_edges = self._correlogram_(lag_max, neuroneA['neurone' + str(idx)].neurones[
                        segment].spike_times_rmz, neuroneB['neurone' + str(idy)].neurones[segment].spike_times_rmz, range_bin, nb_bin)
                    ax[idx][idy].hist(bin_edges[:-1], bin_edges, weights=hist)
                else:
                    ax[idx][idy].set_fc('red')
                    lag, hist, bin_edges = self._correlogram_(lag_max, neuroneA['neurone' + str(idx)].neurones[
                        segment].spike_times_rmz,neuroneB['neurone' + str(idy)].neurones[segment].spike_times_rmz, range_bin, nb_bin)
                    ax[idx][idy].hist(bin_edges[:-1], bin_edges, weights=hist)
                    ax[idx][idy].set_xlabel('neurone' + str(idx) + ' : ' + 'neurone' + str(idx))
                    # ax[idx][idy].set_ylabel('Fréquence (Hz)')

        self._save_figure_(fig=fig1, name=name, option='crosscorrelogramme')

    def plot_frequence_glissante(self, segment: int, taille_fenetre: int, pas_de_gliss: int, name: str,
                                 choix_neurone: List[str] = None):
        fig1 = self._crea_fig_()
        neuroneA: Dict[str, SpikeSegment] = self.neurone
        items_neurone, items_segment, items_voltage_signal = self._items_(segment, choix_neurone)

        axis_list = self._axis_list_(items_neurone, fig1)

        for ind_subplot, ax in enumerate(axis_list):
            freq_all = neuroneA[items_neurone[ind_subplot]].neurones[segment].spike_times_rmz.size / \
                       neuroneA[items_neurone[ind_subplot]].neurones[segment].spike_times_rmz[-1]
            freq, nb_bin = self._calcul_frequence_glissant_(
                neuroneA[items_neurone[ind_subplot]].neurones[segment].spike_times_rmz, taille_fenetre, pas_de_gliss)
            ax.plot(nb_bin[0], freq)
            ax.set_xlabel(items_neurone[ind_subplot] + 'fréq mean : ' + str(freq_all))
            ax.set_ylabel('Fréquence (Hz)')

        self._save_figure_(fig=fig1, name=name, option='frequence')

    def plot_burst_glissant(self, segment: int, taille_fenetre: int, pas_de_gliss: int, name: str,
                            choix_neurone: List[str] = None):

        neuroneA: Dict[str, SpikeSegment] = self.neurone
        items_neurone, items_segment, items_voltage_signal = self._items_(segment, choix_neurone)

        fig1 = self._crea_fig_()
        axis_list = self._axis_list_(items_neurone, fig1)

        for ind_subplot, ax in enumerate(axis_list):
            nb_swb, nb_bin = self._calcul_swb_glissant_(
                neuroneA[items_neurone[ind_subplot]].neurones[segment].spike_times_rmz,
                neuroneA[items_neurone[ind_subplot]].neurones[segment].spike_times_isi,
                taille_fenetre, pas_de_gliss)

            ax.plot(nb_bin[0], nb_swb)
            ax.set(ylim=(0,100))
            ax.set_xlabel(items_neurone[ind_subplot] + 'swb mean : ' + str(
                neuroneA[items_neurone[ind_subplot]].neurones[segment].spike_times_swb))
            ax.set_ylabel('SWB (%)')

        self._save_figure_(fig=fig1, name=name, option='swbmean')

    def _event_timming_(self, event: EventFileNeuralynx, items_segment: List[str], intervalle: List[int]):

        items_event_in_segment_start: List[str] = [i for i in event.event[items_segment[0].split(',')[0]].keys() if i.find('start') != -1]
        items_event_in_segment_on: List[str] = [i for i in event.event[items_segment[0].split(',')[0]].keys() if i.find('start') == -1]

        items_event_in_segment_point_in_vector_time_start: List[str] = [i for i in event.event[items_segment[0].split(',')[0]][items_event_in_segment_start[0]].keys() if i.find('recaler') != -1]
        event_time_in_raw_all: ndarray = event.event[items_segment[0].split(',')[0]][items_event_in_segment_start[0]][items_event_in_segment_point_in_vector_time_start[0]]
        event_time_in_raw: ndarray = event_time_in_raw_all[((event_time_in_raw_all > intervalle[0]) & (event_time_in_raw_all < intervalle[1]))]

        if not items_event_in_segment_on:
            event_time_in_raw_on: ndarray = event_time_in_raw
        else:
            items_event_in_segment_point_in_vector_time_on: List[str] = [i for i in
                                                                         event.event[items_segment[0].split(',')[0]][
                                                                             items_event_in_segment_on[0]].keys() if
                                                                         i.find('recaler') != -1]
            event_time_in_raw_all_on: ndarray = \
            event.event[items_segment[0].split(',')[0]][items_event_in_segment_start[0]][
                items_event_in_segment_point_in_vector_time_on[0]]

            event_time_in_raw_on: ndarray = event_time_in_raw_all_on[
                ((event_time_in_raw_all_on > intervalle[0]) & (event_time_in_raw_all_on < intervalle[1]))]


        return event_time_in_raw, event_time_in_raw_on


    def plot_neurone_in_raw(self, intervalle: List[int], neurone: Dict[str, SpikeSegment], event: EventFileNeuralynx,
                            segment: int, choix_neurone: List[str] = None, name: str = None, option_csc: List[str] = None):
        """
                Permet de plot un ou plusieurs neurones sur la/les trace brute choisi dans option_csc.
                Si aucun csc est choisi (option_csc), toutes les traces contenues dans le dossier seront plotées
                Si aucun neurone est choisi (choix_neurone), ils seront tous plotés

        :param intervalle:
        :param neurone:
        :param segment:
        :param choix_neurone:
        :param name:
        :param option_csc:
        :return:
        """
        colors = ['red', 'b', 'black', 'g', 'red', 'b', 'black', 'g']

        fig1 = self._crea_fig_()

        items_neurone, items_segment, items_voltage_signal = self._items_(segment, choix_neurone, option_csc)
        axis_list = self._axis_list_(items_voltage_signal, fig1)

        voltage_signals: Dict[str, RawSignal] = self.neurone[items_neurone[0]].signal.signal_segments[
            items_segment[segment]].voltage_signals


        event_trace: ndarray([int]) = np.ones([neurone[items_neurone[0]].basetime[items_segment[0].split(',')[0]].size], dtype=int) * 300

        event_time_in_raw, event_time_in_raw_on = self._event_timming_(event, items_segment, intervalle)

        name_neurone: str
        for idx, name_neurone in enumerate(items_neurone):
            base_time: ndarray = neurone[name_neurone].neurones[segment].basetime
            spike_time_in_raw = neurone[name_neurone].neurones[segment].spike_time_in_raw[((neurone[name_neurone].neurones[segment].spike_time_in_raw >
                                intervalle[0]) &(neurone[name_neurone].neurones[segment].spike_time_in_raw <intervalle[1]))]
            for ind_subplot, ax in enumerate(axis_list):
                ax.plot(base_time[intervalle[0]:intervalle[1]],
                        voltage_signals[items_voltage_signal[ind_subplot]].magnitude[intervalle[0]:intervalle[1]],
                        'black', linewidth=0.1)

                ax = self._insertspikeinplotraw_(ax, base_time, items_voltage_signal, ind_subplot, spike_time_in_raw,
                                                 voltage_signals, colors[idx])

                ax = self._inserteventinplotraw_(ax, base_time, items_voltage_signal, ind_subplot, event_time_in_raw,
                                                 voltage_signals, event_trace)
                ax = self._inserteventinplotraw_(ax, base_time, items_voltage_signal, ind_subplot, event_time_in_raw_on,
                                                 voltage_signals, event_trace)
                ax.set_xlabel(voltage_signals[items_voltage_signal[ind_subplot]].info_channels)
                ax.set_ylabel('µV')

        self._save_figure_(fig=fig1, name=name, option='allspike')

    def plot_first_spike_in_swb_neurone_in_raw(self, intervalle: List[int], neurone: Dict[str, SpikeSegment],
                                               event: EventFileNeuralynx, segment: int,
                                               name: str = None, choix_neurone: List[str] = None,
                                               option_csc: List[str] = None):

        colors = ['red', 'b', 'black', 'g', 'red', 'b', 'black', 'g']

        fig1 = self._crea_fig_()
        items_neurone, items_segment, items_voltage_signal = self._items_(segment, choix_neurone, option_csc)
        axis_list = self._axis_list_(items_voltage_signal, fig1)
        voltage_signals: Dict[str, RawSignal] = self.neurone[items_neurone[0]].signal.signal_segments[
            items_segment[segment]].voltage_signals


        event_trace: ndarray([int]) = np.ones([neurone[items_neurone[0]].basetime[items_segment[0].split(',')[0]].size], dtype=int) * 300

        event_time_in_raw, event_time_in_raw_on = self._event_timming_(event, items_segment, intervalle)


        name_neurone: str
        for idx, name_neurone in enumerate(items_neurone):
            base_time: ndarray = neurone[name_neurone].neurones[segment].basetime

            # extraction des premiers spikes des burst comprisent  uniquement dans les events
            spike_time_in_r = neurone[name_neurone].neurones[segment].spike_time_in_raw[
                neurone[name_neurone].neurones[segment].all_info_swb['idx_swb_start']]

            spike_time_in_raw = spike_time_in_r[((spike_time_in_r > intervalle[0]) & (spike_time_in_r < intervalle[1]))]

            for ind_subplot, ax in enumerate(axis_list):
                ax.plot(base_time[intervalle[0]:intervalle[1]],
                        voltage_signals[items_voltage_signal[ind_subplot]].magnitude[intervalle[0]:intervalle[1]],
                        'black', linewidth=0.1)

                ax = self._insertspikeinplotraw_(ax, base_time, items_voltage_signal, ind_subplot, spike_time_in_raw,
                                                 voltage_signals, colors[idx])

                ax = self._inserteventinplotraw_(ax, base_time, items_voltage_signal, ind_subplot, event_time_in_raw,
                                                 voltage_signals, event_trace)
                ax = self._inserteventinplotraw_(ax, base_time, items_voltage_signal, ind_subplot, event_time_in_raw_on,
                                                 voltage_signals, event_trace)
                ax.set_xlabel(voltage_signals[items_voltage_signal[ind_subplot]].info_channels)
                ax.set_ylabel('µV')

        self._save_figure_(fig=fig1, name=name, option='firstspike')

    def plot_waveform_neurone(self, neurone: Dict[str, SpikeSegment], segment: int,
                              intervalle: List[int] = None, name: str = None, choix_neurone: List[str] = None,
                              option_csc: List[str] = None):
        colors = ['red', 'b', 'g', 'y', 'black']

        fig1 = self._crea_fig_()
        items_neurone, items_segment, items_voltage_signal = self._items_(segment, choix_neurone, option_csc)

        axis_list = self._axis_list_(items_voltage_signal, fig1)

        voltage_signals: Dict[str, RawSignal] = self.neurone[items_neurone[0]].signal.signal_segments[
            items_segment[segment]].voltage_signals
        if not intervalle:
            intervalle: List[int] = [0, len(voltage_signals[[i for i in voltage_signals.keys()][0]].magnitude)]


        x_spike = np.arange(0, 50)
        name_neurone: str
        for idx, name_neurone in enumerate(items_neurone):
            spike_time_in_raw = neurone[name_neurone].neurones[segment].spike_time_in_raw[((
                         neurone[name_neurone].neurones[segment].spike_time_in_raw > intervalle[0]) &
                            (neurone[name_neurone].neurones[segment].spike_time_in_raw < intervalle[1]))]

            for ind_subplot, ax in enumerate(axis_list):
                for i in spike_time_in_raw:
                    ax.plot(x_spike, voltage_signals[items_voltage_signal[ind_subplot]].magnitude[
                                     i - self._number_of_point:i + self._number_of_point],
                            colors[idx], linewidth=self._thickness)
                ax.set_xlabel(voltage_signals[items_voltage_signal[ind_subplot]].info_channels)
                ax.set_ylabel('µV')

        self._save_figure_(fig=fig1, name=name, option='waveform')


    def plot_raster_event_spike(self, neurone: Dict[str, SpikeSegment], event: EventFileNeuralynx, segment: int,
                                choix_neurone: List[str] = None, name: str = None):
        """
        normalisation entre -1 et 1 ! faire attention car si la fenetre est dif de 1 la base temporelle restara entre -1 et 1
        (2*(e.loc[3][e.loc[3].notna()]-(time_event[3] - fenetre))/((time_event[3] + fenetre)-(time_event[3] - fenetre)))-1
        """
        fig1 = self._crea_fig_()
        # ee = self._items_(segment, choix_neurone)
        items_neurone, items_segment, items_voltage_signal = self._items_(segment, choix_neurone)
        spike_time = neurone['neurone0'].neurones[1].spike_times_rmz

        items_event_by_segment = list(event.event.keys())
        items_event_in_segment = list(event.event[items_event_by_segment[1]].keys())
        event_in_segment: Dict = event.event['num segment : 1']['num segment : 1, num event start : 0']
        items_event_time_in_segment = list(event_in_segment.keys())

        time_event: Quantity = event_in_segment['event brute'][event_in_segment['reward index']]
        fenetre = Quantity(1, 's')
        # spike_time[((spike_time < time_event[2] + fenetre) & (spike_time > time_event[2] - fenetre))]

        axis_list = self._axis_list_(items_neurone, fig1)
        lineoffsets1 = np.array([1])
        linelengths1 = [1.5]
        lineoffsets2 = 1
        linelengths2 = 1
        #
        # colors1 = 'black'
        colors2 = 'r'
        e = pd.DataFrame()
        _tmp = pd.Series(dtype=float)
        for id, val in enumerate(time_event):
            # spike_around = pd.Series(
            #     spike_time[((spike_time < time_event[id] + fenetre) & (spike_time > time_event[id] - fenetre))], name=str(id))
            e = e.append(pd.Series(
                spike_time[((spike_time < time_event[id] + fenetre) & (spike_time > time_event[id] - fenetre))], name=str(id)))

        val: str
        # idx: int
        for ind_subplot, ax in enumerate(axis_list):
            for idx, val in enumerate(time_event):
                _spike_around_norma = (2*(e.iloc[idx][e.iloc[idx].notna()]-(time_event[idx] - fenetre))/((time_event[idx] + fenetre)-(time_event[idx] - fenetre)))-1
                _tmp = _tmp.append(_spike_around_norma)
                ax.eventplot(_spike_around_norma,linelengths=linelengths2, lineoffsets=lineoffsets2+idx, colors=colors2)
                # ax.eventplot(e.loc[2][e.loc[2].notna()],linelengths=linelengths2, lineoffsets=lineoffsets2, colors=colors2)

        print('rr')
        self._save_figure_(fig=fig1, name=name, option='raster')
        return _tmp

    def plot_raster_event_spikev2(self, spike_time: ndarray, time_event: Quantity, event: EventFileNeuralynx, segment: int,
                                choix_neurone: List[str] = None, name: str = None):
        """
        normalisation entre -1 et 1 ! faire attention car si la fenetre est dif de 1 la base temporelle restara entre -1 et 1
        (2*(e.loc[3][e.loc[3].notna()]-(time_event[3] - fenetre))/((time_event[3] + fenetre)-(time_event[3] - fenetre)))-1
        """
        fig1 = self._crea_fig_()
        # ee = self._items_(segment, choix_neurone)
        items_neurone, items_segment, items_voltage_signal = self._items_(segment, choix_neurone)


        # items_event_by_segment = list(event.event.keys())
        # items_event_in_segment = list(event.event[items_event_by_segment[1]].keys())
        event_in_segment: Dict = event.event['num segment : 1']['num segment : 1, num event start : 0']
        # items_event_time_in_segment = list(event_in_segment.keys())

        time_event: Quantity = event_in_segment['event brute'][event_in_segment['reward index']]
        fenetre = Quantity(1, 's')

        axis_list = self._axis_list_(items_neurone, fig1)
        lineoffsets1 = np.array([1])
        linelengths1 = [1.5]
        lineoffsets2 = 1
        linelengths2 = 1
        #
        # colors1 = 'black'
        colors2 = 'r'
        e = pd.DataFrame()
        _tmp = pd.Series(dtype=float)
        for id, val in enumerate(time_event):
            # spike_around = pd.Series(
            #     spike_time[((spike_time < time_event[id] + fenetre) & (spike_time > time_event[id] - fenetre))], name=str(id))
            e = e.append(pd.Series(
                spike_time[((spike_time < time_event[id] + fenetre) & (spike_time > time_event[id] - fenetre))], name=str(id)))

        val: str
        # idx: int
        for ind_subplot, ax in enumerate(axis_list):
            for idx, val in enumerate(time_event):
                _spike_around_norma = (2*(e.iloc[idx][e.iloc[idx].notna()]-(time_event[idx] - fenetre))/((time_event[idx] + fenetre)-(time_event[idx] - fenetre)))-1
                _tmp = _tmp.append(_spike_around_norma)
                ax.eventplot(_spike_around_norma, linelengths=linelengths2, lineoffsets=lineoffsets2+idx, colors=colors2)
                # ax.eventplot(e.loc[2][e.loc[2].notna()],linelengths=linelengths2, lineoffsets=lineoffsets2, colors=colors2)

        print('rr')
        self._save_figure_(fig=fig1, name=name, option='raster')
        return _tmp

    def plot_kernel_density(self, _tmp, name: str):
        fig1 = self._crea_fig_()

        Vecvalues = _tmp[:, None]
        Vecpoints = np.linspace(-1, 1, 100)[:, None]
        kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(Vecvalues)
        logkde = kde.score_samples(Vecpoints)
        plt.plot(Vecpoints, np.exp(logkde))
        self._save_figure_(fig=fig1, name=name, option='kdensity')


if __name__ == "__main__":
    t1 = chrono.time()
    # csc_dir: str = r'D:\Dropbox\python\import_neuralynxv2\data\pinp2_bsl 16012020'

    csc_dir: str = r'/data/pinp8 06022020'
    # csc_dir: str = r'D:\Dropbox\python\import_neuralynxv2\data\cplx07 + bsl'
    # csc_dir: str = r'D:\Dropbox\python\import_neuralynxv2\data\equequ1 - test Steve'

    ext: str = '/*.txt'
    spike_file_folder: str = r'D:\Dropbox\python\import_neuralynxv2\data\pinp8 06022020\clustering\Nouveau_dossier' + ext
    # spike_file_folder: str = r'D:\Dropbox\python\import_neuralynxv2\data\cplx07 + bsl\clustering' + ext
    # spike_file_folder: str = r'D:\Dropbox\python\import_neuralynxv2\data\pinp2_bsl 16012020\clustering pos11000' + ext
    # spike_file_folder: str = r'D:\Dropbox\python\import_neuralynxv2\data\equequ1 - test Steve\clustering' + ext


    spike_files = LoadSpikeFiles(spike_file_folder, csc_dir)

    neurones = spike_files.neurone()

    Event = EventFileNeuralynx(csc_dir)
    Event.make_event()


    plot = GenericPlot(neurones)
    # plot.plot_event(Event,110,120)

    #
    # raster = plot.plot_raster_event_spike(neurone=neurones, event=Event, segment=0, choix_neurone=['neurone0'], name='neurone0')
    # plot.plot_kernel_density(_tmp=raster,name='neurone0')
    #
    # # intervalle = [neurones['neurone0'].neurones[0].event_time_in_raw[6], neurones['neurone0'].neurones[0].event_time_in_raw[15]]
    intervalle = [163 * 32000, 164 * 32000]  # intervalle par seconde
    # intervalle = [5216000, 5222000] # intervalle par point
    #
    plot.plot_neurone_in_raw(intervalle=intervalle, neurone=neurones, event=Event, segment=0, choix_neurone=['neurone0'],
                                    name='test', option_csc=['CSC2', 'CSC6', 'CSC7', 'CSC14'])
    #
    # plot.plot_neurone_in_raw(intervalle=intervalle, neurone=neurones, event=Event, segment=0,
    #                                 name='test', option_csc=['CSC2', 'CSC6', 'CSC7', 'CSC10'])
    #
    # plot.plot_first_spike_in_swb_neurone_in_raw(intervalle=intervalle, neurone=neurones, event=Event, segment=0, name='test',
    #                   choix_neurone=['neurone0'], option_csc=['CSC2', 'CSC6', 'CSC7', 'CSC10'])
    #
    # plot.plot_frequence_glissante(segment=0, taille_fenetre=30, pas_de_gliss=5, name='freq slide')
    #
    plot.plot_burst_glissant(segment=0, taille_fenetre=30, pas_de_gliss=5, name='burst slide')
    #
    # plot.plot_waveform_neurone(intervalle=intervalle, neurone=neurones, segment=0, name='waveform',
    #                    choix_neurone=['neurone0'], option_csc=['CSC2', 'CSC6', 'CSC7', 'CSC10'])

    # plot.plot_waveform_neurone(neurone=neurones, segment=0, intervalle=intervalle, name='waveform',
    #                     option_csc=['CSC1', 'CSC2', 'CSC3', 'CSC6'])

    # plot.plot_waveform_neurone(neurone=neurones, segment=0, name='waveform',
    #                     option_csc=['CSC1', 'CSC2', 'CSC3', 'CSC6'])
    #
    # plot.plotcorrelogram(lag_max=1, lenght_of_bin=0.005, segment=0, name='cross')
    t2 = chrono.time()

    print(f'temps global : {t2 - t1}')
    # jj = Event.all.diff()

    # =================================================
    """
    pour save tes temps de spike :
    - lorque tu as lancé le code tu reviens dans le terminal python et tu utilise cette ligne
    """
    # SavingMethodes.save_data_text(neurones['neurone0'].neurones[0].spike_times_rmz, 'prout', r'D:\'')
