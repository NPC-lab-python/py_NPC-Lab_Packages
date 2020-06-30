from neo import AnalogSignal
from neo.core.event import Event
from neo.core.block import Block
from neo.io.neuralynxio import NeuralynxIO
import joblib

"""
Création de la classe compléte

"""
from abc import ABC, abstractmethod
from typing import List, Dict, Union
import numpy as np
from numpy.core.multiarray import ndarray
import time as chrono
from quantities.quantity import Quantity

"""
Prend en compte que les fichiers 'nse' 'ncs' 'nev' 'ntt'
pour les fichiers 'nst', il faut passer par un fichier txt de SpikeSort3D issu de SpikeExtractor

IMPORTANT :
Pour le moment, seul les fichiers '.ncs' et '.nev' sont importés
"""


class GlobalRawSignal(object):
    def __init__(self):
        self.base_time: Quantity = None
        self.voltage_signals: Dict[str, RawSignal] = {}


class GlobalEventSignal(object):
    def __init__(self):
        self.event_signals: RawEvent = None


class GlobalEventBasetime(object):
    def __init__(self):
        self.base_time: Quantity = None


Event_type = Dict[str, Union[GlobalEventSignal, GlobalEventBasetime]]


class RawSignals(object):
    """ Utilise pour l'ensemble des signaux

    exemple de typing ; csc_dir: Path signifie que csc_dir doit être de type Path

    tout est contenue dans voltage signal

    Pour plusieurs start/stop dans un même fichier, toutes les datas seront dans différents
    segments avec les events qui correspondent

    """
    Sampling = 32000

    def __init__(self, csc_dir: str):
        self.signal_segments: Dict[str, GlobalRawSignal] = {}
        # self.voltage_signals: List[int, RawSignal] = []
        self.Event: Event_type = {}
        self.csc_dir: str = csc_dir

    def import_event(self, reader: NeuralynxIO, block: Block, Event: Event_type, segments: int = None):
        t1 = chrono.time()
        print(f'Début d"importation des fichiers Event')
        segmentation: list = ['Starting Recording', 'Stopping Recording']

        if not segments:
            segment = 0
            print(f'Aucun segment présent ')

            for i in range(0, len(reader.header['event_channels']), 1):
                print(f'Numero de l"Event : {i + segment}')
                tmp_event = RawEvent()
                tmp_event.global_t_start = reader.global_t_start
                tmp_event.global_t_stop = reader.global_t_stop
                tmp_event.infos = reader.header['event_channels'][i][0]
                tmp_event._event_brut = block.segments[segment].events[i]
                tmp_event._convert_labels_(block.segments[segment].events[i].labels)
                tmp_event.event_each_times = block.segments[segment].events[i].times

                print(
                    f'type de label : {type(tmp_event.event_each_label)} et de times : {type(tmp_event.event_each_times)}')

                if tmp_event.event_each_label.size != 0 and tmp_event.event_each_times.size != 0:

                    if tmp_event.event_each_label[0].find('Starting') != -1:
                        tmp_event.start_delay = [tmp_event.event_each_label[0],
                                                 block.segments[segment].events[i].times[0]]
                        temp_st_sp = [tmp_event.event_each_label == i for i in segmentation]

                        tmp_event.start_stop_segment = [
                            [tmp_event.event_each_times[i], tmp_event.event_each_label[i]]
                            for i in temp_st_sp]
                        # print("Label ok")
                        global_event = GlobalEventSignal()
                        global_event.event_signals = tmp_event
                        Event[f'num segment : {segment}, num event start : {i}'] = global_event
                        Event[f'num segment : {segment}'] = GlobalEventBasetime()

                    else:
                        # print("Label ok")
                        global_event = GlobalEventSignal()
                        global_event.event_signals = tmp_event
                        Event[f'num segment : {segment}, num event : {i}'] = global_event
                        Event[f'num segment : {segment}'] = GlobalEventBasetime()

                else:
                    print("Ne contient aucun élément de Starting")

        else:
            for segment in range(segments):
                print(f'Numero du Segment : {segment}')
                for i in range(0, len(reader.header['event_channels']), 1):
                    print(f'Numero de l"Event : {i + segment}')
                    tmp_event = RawEvent()
                    tmp_event.global_t_start = reader.global_t_start
                    tmp_event.global_t_stop = reader.global_t_stop
                    tmp_event.infos = reader.header['event_channels'][i][0]
                    tmp_event._event_brut = block.segments[segment].events[i]
                    tmp_event._convert_labels_(block.segments[segment].events[i].labels)
                    tmp_event.event_each_times: Quantity = block.segments[segment].events[i].times
                    if tmp_event.event_each_label.size != 0 and tmp_event.event_each_times.size != 0:
                        if tmp_event.event_each_label[0].find('Starting') != -1:
                            tmp_event.start_delay = [tmp_event.event_each_label[0],
                                                     block.segments[segment].events[i].times[0]]
                            temp_st_sp = [tmp_event.event_each_label == i for i in segmentation]
                            tmp_event.start_stop_segment = [
                                [tmp_event.event_each_times[i], tmp_event.event_each_label[i]] for i in
                                temp_st_sp]
                            # print("Label ok")
                            global_event = GlobalEventSignal()
                            global_event.event_signals = tmp_event
                            Event[f'num segment : {segment}, num event start : {i}'] = global_event
                            Event[f'num segment : {segment}'] = GlobalEventBasetime()

                        else:
                            # print("Label ok")
                            global_event = GlobalEventSignal()
                            global_event.event_signals = tmp_event
                            Event[f'num segment : {segment}, num event : {i}'] = global_event
                            Event[f'num segment : {segment}'] = GlobalEventBasetime()

                    else:
                        print("Ne contient aucun élément de Starting. ")
        t2 = chrono.time()
        print(f'Fin d"importation des fichiers Event : {t2 - t1}')
        return Event

    def import_event_light(self, reader: NeuralynxIO, block: Block, segments: int = None):
        t1 = chrono.time()
        print(f'Début d"importation des fichiers Event')
        segmentation: list = ['Starting Recording', 'Stopping Recording']

        if not segments:
            segment = 0
            print(f'Aucun segment présent ')

            for i in range(0, len(reader.header['event_channels']), 1):
                print(f'Numero de l"Event : {i + segment}')
                tmp_event = RawEvent()
                tmp_event.global_t_start = reader.global_t_start
                tmp_event.global_t_stop = reader.global_t_stop
                tmp_event._convert_labels_(block.segments[segment].events[i].labels)
                tmp_event.event_each_times = block.segments[segment].events[i].times

                if tmp_event.event_each_label.size != 0 and tmp_event.event_each_times.size != 0:

                    if tmp_event.event_each_label[0].find('Starting') != -1:

                        temp_st_sp = [tmp_event.event_each_label == i for i in segmentation]

                        tmp_event.start_stop_segment = [
                            [tmp_event.event_each_times[i], tmp_event.event_each_label[i]]
                            for i in temp_st_sp]
                        global_event = GlobalEventSignal()
                        global_event.event_signals = tmp_event
                        self.Event[f'num segment : {segment}, num event start : {i}'] = global_event
                    else:
                        global_event = GlobalEventSignal()
                        global_event.event_signals = tmp_event
                        self.Event[f'num segment : {segment}, num event : {i}'] = global_event
                else:
                    print("Ne contient aucun élément de Starting")

        else:
            for segment in range(segments):
                print(f'Numero du Segment : {segment}')
                for i in range(0, len(reader.header['event_channels']), 1):
                    print(f'Numero de l"Event : {i + segment}')
                    tmp_event = RawEvent()
                    tmp_event.global_t_start = reader.global_t_start
                    tmp_event.global_t_stop = reader.global_t_stop
                    tmp_event._convert_labels_(block.segments[segment].events[i].labels)
                    tmp_event.event_each_times = block.segments[segment].events[i].times

                    if tmp_event.event_each_label.size != 0 and tmp_event.event_each_times.size != 0:
                        if tmp_event.event_each_label[0].find('Starting') != -1:
                            temp_st_sp = [tmp_event.event_each_label == i for i in segmentation]
                            tmp_event.start_stop_segment = [
                                [tmp_event.event_each_times[i], tmp_event.event_each_label[i]] for i in
                                temp_st_sp]
                            global_event = GlobalEventSignal()
                            global_event.event_signals = tmp_event
                            self.Event[f'num segment : {segment}, num event start : {i}'] = global_event
                        else:
                            global_event = GlobalEventSignal()
                            global_event.event_signals = tmp_event
                            self.Event[f'num segment : {segment}, num event : {i}'] = global_event

                    else:
                        print("Ne contient aucun élément de Starting. ")
        t2 = chrono.time()
        print(f'Fin d"importation des fichiers Event : {t2 - t1}')

    def import_csc(self, reader: NeuralynxIO, block: Block, segments: int = None):
        t1 = chrono.time()
        print(f'Début d"importation des fichiers CSC')

        if not segments:
            segment = 0
            i: int
            global_signal = GlobalRawSignal()
            for i in range(0, len(reader.header['signal_channels']), 1):
                tmp_signal = RawSignal()
                tmp_signal.info_channels = reader.header['signal_channels'][i][0]
                tmp_signal.num_channel = i
                tmp_signal._signal_brut = block.segments[segment].analogsignals[i]
                # tmp_signal.time = block.segments[segment].analogsignals[i].times
                tmp_signal.magnitude = block.segments[segment].analogsignals[i].magnitude
                # self.voltage_signals.append(tmp_signal)
                global_signal.voltage_signals[tmp_signal.info_channels] = tmp_signal

            global_signal.base_time: Quantity = block.segments[segment].analogsignals[0].times

            self.signal_segments[f'num segment : {segment}, num raw : {i}'] = global_signal

        else:
            for segment in range(segments):
                i: int
                global_signal = GlobalRawSignal()
                for i in range(0, len(reader.header['signal_channels']), 1):
                    tmp_signal = RawSignal()
                    tmp_signal.info_channels = reader.header['signal_channels'][i][0]
                    tmp_signal.num_channel = i
                    tmp_signal._signal_brut = block.segments[segment].analogsignals[i]
                    # tmp_signal.time = block.segments[segment].analogsignals[i].times
                    tmp_signal.magnitude = block.segments[segment].analogsignals[i].magnitude
                    # self.voltage_signals.append(tmp_signal)
                    global_signal.voltage_signals[tmp_signal.info_channels] = tmp_signal
                global_signal.base_time: Quantity = block.segments[segment].analogsignals[0].times

                self.signal_segments[f'num segment : {segment}, num raw : {i}'] = global_signal
        t2 = chrono.time()
        print(f'Fin d"importation des fichiers CSC : {t2 - t1}')

    def import_csc_light(self, block: Block, Event: Event_type, segments: int = None):
        t1 = chrono.time()
        print(f'Début d"importation des fichiers CSC')

        if not segments:
            segment = 0
            # global_signal = GlobalRawSignal()
            #
            # global_signal.base_time = block.segments[segment].analogsignals[0].times
            #
            # self.signal_segments[f'num segment : {segment}, num raw : 0'] = global_signal
            Event[f'num segment : {segment}'].base_time = block.segments[segment].analogsignals[0].times
        else:
            for segment in range(segments):
                # i: int
                # global_signal = GlobalRawSignal()
                #
                # global_signal.base_time = block.segments[segment].analogsignals[0].times
                #
                # self.signal_segments[f'num segment : {segment}, num raw : 0'] = global_signal
                Event[f'num segment : {segment}'].base_time = block.segments[segment].analogsignals[0].times

        t2 = chrono.time()
        print(f'Fin d"importation des fichiers CSC : {t2 - t1}')
        return Event

    def load(self, option: str = None):

        t1 = chrono.time()
        reader = NeuralynxIO(dirname=self.csc_dir, use_cache=False, cache_path=self.csc_dir, )
        block = reader.read_block(signal_group_mode='split-all')
        nb_segment: int = int(len(block.segments))
        if option is None:
            self.import_csc(reader, block, nb_segment)
            self.Event = self.import_event(reader, block, self.Event, nb_segment)
        elif option == 'csc':
            self.import_csc(reader, block, nb_segment)
            self.import_event_light(reader, block, nb_segment)

        elif option == 'event':
            self.Event = self.import_event(reader, block, self.Event, nb_segment)
            self.Event = self.import_csc_light(block, self.Event, nb_segment)
        t2 = chrono.time()
        print(f'temps global d"importation des fichiers : {t2 - t1}')


class RawSignal(object):
    """
    utilisé pour un signal
    """

    def __init__(self):
        self.info_channels: str = None
        self.num_channel: int = None
        self._signal_brut: AnalogSignal = []
        # self.time = None
        self.magnitude = None


class RawEvent(object):
    """
    utilisé pour un event déjà convertie en seconde
    par la lib quantities
    """

    def __init__(self):
        self._event_brut: List[Event] = []
        self.infos: str = None
        self.global_t_start: float = None
        self.global_t_stop: float = None
        self.event_each_label: ndarray[str] = np.array([], dtype=str)
        self.event_each_times: Quantity = Quantity([], 's')
        self.start_delay: List[str, float] = None
        self.start_stop_segment: List[str, float] = None

    def _convert_labels_(self, labels: Block):
        """
        les labels sont dans un format'numpy.bytes_' qu'il faut convertir en 'UTF-8'
        pour pouvoir le tester
        :return:
        """
        # print(f'Vérif des labels : {labels}')
        val: np.byte
        for i, val in enumerate(labels):
            self.event_each_label = np.hstack((self.event_each_label, str(val.decode('UTF-8'))))
            # print(str(val.decode('UTF-8')))

            # self.event_each_label.append(str(val.decode('UTF-8')))


# class PlotData(RawSignals):
#
#     def __init__(self, csc_dir):
#         super().__init__(csc_dir)
#         self.csc_dir: Path = csc_dir
#         self.utilitaire_neuralynx()
#         # self.plotrawsignal()
#
#     def plotrawsignal(self, *args):
#         if len(args) == 0:
#             fig1 = plt.figure(4 * 1000)
#             plt.clf()
#             for ind_subplot in range(0, len(self.voltage_signals)):
#                 print(ind_subplot)
#                 ax = fig1.add_subplot(len(self.voltage_signals), 1, ind_subplot + 1)
#                 ax.plot(self.voltage_signals[ind_subplot].time[0:100000],
#                         self.voltage_signals[ind_subplot].magnitude[0:100000])
#                 ax.set_xlabel(self.voltage_signals[ind_subplot].info_channels)
#                 ax.set_ylabel('µV')
#
#             fig1.show()
#         else:
#             spike_times = args[0]
#             fig1 = plt.figure(4 * 1000)
#             plt.clf()
#             for ind_subplot in range(0, len(self.voltage_signals)):
#                 print(ind_subplot)
#                 ax = fig1.add_subplot(len(self.voltage_signals), 1, ind_subplot + 1)
#                 ax.plot(self.voltage_signals[ind_subplot].time[0:100000],
#                         self.voltage_signals[ind_subplot].magnitude[0:100000])
#                 ax.set_xlabel(self.voltage_signals[ind_subplot].info_channels)
#                 ax.set_ylabel('µV')
#
#             fig1.show()


if __name__ == "__main__":
    csc_dir: str = r'/data/pinp8 06022020'

    reader = NeuralynxIO(dirname=csc_dir, use_cache=False, cache_path=csc_dir)
    block = reader.read_block(signal_group_mode='split-all')
    len(block.segments)
    block.segments[1].events[0]

    tet = RawSignals(csc_dir)
    tet.load()
