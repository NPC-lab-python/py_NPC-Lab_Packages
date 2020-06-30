from py_NPClab_Package.traitement_event.EventTraitement import ClassificationPattern
from pandas import DataFrame, Series
import pandas as pd
import numpy as np
import logging
import sys
logging.basicConfig(level=logging.DEBUG)
# logging.getLogger('matplotlib').setLevel(logging.WARNING)

class AnalyseOpto(ClassificationPattern):

    def set_opto_profile(self, dir_profile_pattern, time_up, time_down, pattern: str, name_profile: str):
        """
        création du fichier profile pattern
        :param name: correspond au nom du fichier (nom du profile)

        """
        profile_pattern = self._set_pattern_(dir_profile_pattern=dir_profile_pattern, time_up=time_up,
                           time_down=time_down, pattern=pattern, name=name_profile)



    def get_opto_profile(self, dir_pattern: str, name_pattern: str):
        """
        Permet de charger le profile_pattern
        """
        profile_pattern = self._set_pattern_(dir_profile_pattern=dir_pattern, name=name_pattern)
        return profile_pattern

    def set_opto_from_profile(self, profile_pattern: dict, reward_time: Series, reward_index: Series):
        """

        """
        all_info_pattern = pd.DataFrame([])
        start_stop, time_structure_pattern = self._pattern_event_(time_down=profile_pattern['time_down'],
                                          time_up=profile_pattern['time_up'], name=profile_pattern['name'],
                                          reward_time=reward_time, pattern=profile_pattern['pattern'])
        try:
            assert len(start_stop) > 1
        except AssertionError:
            logging.debug(f'Probléme au niveau de la détection du pattern!!!!')
            sys.exit()
        logging.debug(f'Détection de pattern réussi, nb de pattern : {len(start_stop)}!!!!')

        self.start = pd.Series([reward_time[start_stop[i][0]] for i in range(len(start_stop))])
        self.start_index = pd.Series([reward_index[start_stop[i][0]] for i in range(len(start_stop))])

        all_info_pattern['start'] = pd.Series([reward_time[start_stop[i][0]] for i in range(len(start_stop))])
        all_info_pattern['stop'] = pd.Series([reward_time[start_stop[i][1]] for i in range(len(start_stop))])
        all_info_pattern['start_index'] = pd.Series([reward_index[start_stop[i][0]] for i in range(len(start_stop))])
        all_info_pattern['stop_index'] = pd.Series([reward_index[start_stop[i][1]] for i in range(len(start_stop))])
        all_info_pattern['profile_pattern'] = pd.Series([profile_pattern])
        all_info_pattern['time_structure'] = pd.Series([{'TS':time_structure_pattern}])


        return self.start, self.start_index, all_info_pattern

    def time_pattern(self, start_stop_time: DataFrame, reward_time: Series):
        """
        "start_stop_time" correspond à  "all_info"
        """
        rr = pd.DataFrame([], columns=pd.RangeIndex(0,
                        len(reward_time[(reward_time >= start_stop_time['start'][0]) & (reward_time < start_stop_time['stop'][0])])),
                          index=pd.RangeIndex(0, len(start_stop_time.index)+1))
        z = np.array([])
        for i in range(len(start_stop_time)):
           z = np.append(z, [np.array(reward_time[(reward_time >= start_stop_time['start'][i]) & (reward_time < start_stop_time['stop'][i])])])
           rr.loc[i] = np.array(reward_time[(reward_time >= start_stop_time['start'][i]) & (reward_time < start_stop_time['stop'][i])])
        rr.loc[rr.index[-1]] = np.array(start_stop_time['time_structure'][0]['TS'][(reward_time >= start_stop_time['start'][0]) & (reward_time < start_stop_time['stop'][0])])
        return rr

    def extract_time_spike(self, neurone, time_pattern):
        # data_brute = pd.DataFrame([])
        z = np.array([])
        zs = np.array([])
        a = list(range(1, len(time_pattern.columns), 2))
        m = list(range(0, len(time_pattern.columns), 2))
        k = list(range(2, len(time_pattern.columns), 2))
        l = list(range(1, len(time_pattern.columns)-1, 2))
        spike_time_down = pd.Series([], dtype=float)
        spike_time_up = pd.Series([], dtype=float)

        for idx in range(len(time_pattern.index)-1):
            for i in range(len(a)):
                # print(f'{m[i]},{a[i]}')
                z = np.hstack((z, np.array(neurone.data['time'][(neurone.data['time'] >= time_pattern.loc[idx, m[i]]) & (neurone.data['time'] <= time_pattern.loc[idx, a[i]])]-time_pattern.loc[idx, m[i]])))
                spike_time_up = spike_time_up.append(neurone.data['time'][(neurone.data['time'] >= time_pattern.loc[idx, m[i]]) & (neurone.data['time'] <= time_pattern.loc[idx, a[i]])]-time_pattern.loc[idx, m[i]])

            for i in range(len(k)):
                # print(f'{m[i]},{a[i]}')
                zs = np.hstack((zs, np.array(neurone.data['time'][(neurone.data['time'] >= time_pattern.loc[idx, l[i]]) & (neurone.data['time'] <= time_pattern.loc[idx, k[i]])]-time_pattern.loc[idx, l[i]])))
                spike_time_down = spike_time_down.append(neurone.data['time'][(neurone.data['time'] >= time_pattern.loc[idx, l[i]]) & (neurone.data['time'] <= time_pattern.loc[idx, k[i]])]-time_pattern.loc[idx, l[i]])
        val_time_down, bin_time_down = np.histogram(zs)
        val_time_up, bin_time_up = np.histogram(z)
        data_brute = [spike_time_down, spike_time_up]

        # data_brute['spike_time_down'] = spike_time_down
        # data_brute['spike_time_up'] = spike_time_up


        return val_time_up, bin_time_up, val_time_down, bin_time_down, data_brute

if __name__ == "__main__":
    # -------------------------------------------- partie event ---------------------------------------------------

    from py_NPClab_Package.utilitaire_load.basic_load import ImportNeuralynx, LoadData
    from py_NPClab_Package.traitement_event.EventTraitement import EventFileNeuralynx, EventRewardNeuralynx

    dir_data: str = r'Y:\python\import_neuralynxv2\data\pinp8 06022020'

    event_brute = LoadData.init_data(ImportNeuralynx, dir_data, 'event')

    Event = EventFileNeuralynx(dir_data=dir_data, event=event_brute)
    info_event, all_event = Event.set_event()

    # -------------------------------------------- partie reload specifique neurone ---------------------------------

    from py_NPClab_Package.utilitaire_load.basic_load import NeuroneFilesSerialiser

    dir_save = r'Y:\python\import_neuralynxv2\data\pinp8 06022020\save'
    name = 'segment0_neurone0'
    neurone = LoadData.init_data(NeuroneFilesSerialiser, dir_save, name)

    # ------------------------------------- parti reload specification segment ----------------------------------

    from py_NPClab_Package.utilitaire_load.basic_load import SegmentFilesSerialiser

    dir_save = r'Y:\python\import_neuralynxv2\data\pinp8 06022020\save'
    name = 'segment_infos'
    num_segment = 0
    base_time_segment = LoadData.init_data(SegmentFilesSerialiser, dir_save, name, num_segment)
    # --------------------------------------
    raw_brute = LoadData.init_data(ImportNeuralynx, dir_data, 'csc')


    # reward = EventRewardNeuralynx()

    # Si le dossier "profile_pattern" n'existe pas , il sera créé.
    # reward.set_profile(dir_profile_pattern=dir_data, time_down=0.0475, time_up=0.0025,
    #                    pattern='CABABABABA', name_profile='synchro_video_classic')
    # reward.set_profile(dir_profile_pattern=dir_data, time_down=0.005, time_up=0.005,
    #                    pattern='AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA', name_profile='stimulation_classic')
    # reward.creat_pattern(dir_profile_pattern=dir_data, time_down=0.9, time_up=0.1,
    #                      pattern='ABABABABABABABABABA', name='protocole_opto_1')
    #
    # start_stim, start_stim_index = reward.set_reward(reward_time=all_event['num segment : 0']['time'],
    #                                                  reward_index=all_event['num segment : 0']['index'])
    # start_stim_index = start_stim_index.astype(dtype=int)

    dir_pattern = r'Y:\python\import_neuralynxv2\data\pinp8 06022020\profile_pattern'
    pattern = AnalyseOpto()
    pattern.set_opto_profile(dir_profile_pattern=dir_data, time_down=0.09, time_up=0.01,
                             pattern='CABABABABABABABABABA', name_profile='protocole_opto_10hz')

    # pattern.set_opto_profile(dir_profile_pattern=dir_data, time_down=0.045, time_up=0.005,
    #                          pattern='CABABABABABABABABABA', name_profile='protocole_opto_4')

    profile_pattern = pattern.get_opto_profile(dir_pattern=dir_pattern, name_pattern='protocole_opto_10hz')

    start_stop_time, start_stop_index, all_info = pattern.set_opto_from_profile(profile_pattern=profile_pattern,
                                               reward_time=all_event['num segment : 0']['time'],
                                                reward_index=all_event['num segment : 0']['index'])

    time_pattern = pattern.time_pattern(start_stop_time=all_info, reward_time=all_event['num segment : 0']['time'])
    val_time_up, bin_time_up, val_time_down, bin_time_down, data_brute = pattern.extract_time_spike(neurone=neurone, time_pattern=time_pattern)

    from py_NPClab_Package.utilitaire_plot.BasicPlotSpike import GenericPlotV2

    plot = GenericPlotV2()

    plot.plothist(val_time_up, bin_time_up, ['test'], data_brute)
    # plot.plothist(val_time_down, bin_time_down, ['test'],)

    debut_index = all_event['num segment : 0']['time'][
        all_event['num segment : 0']['time'] == time_pattern.loc[0, time_pattern.columns[0]]].index
    # fin_index = all_event['num segment : 0']['time'][
    #     all_event['num segment : 0']['time'] == time_pattern.loc[time_pattern.index[-2], time_pattern.columns[-1]]].index
    fin_index = all_event['num segment : 0']['time'][
        all_event['num segment : 0']['time'] == time_pattern.loc[time_pattern.index[0], time_pattern.columns[-1]]].index
    intervalle = [all_event['num segment : 0']['index'][debut_index[0]].astype(int),
                  all_event['num segment : 0']['index'][fin_index[0]].astype(int)]  # intervalle par seconde


    plot.plot_neurone_in_raw(intervalle=intervalle,
                             neurones_index_val=[neurone.data['time_index_in raw'][data_brute[1].index]],
                             raw_signal=raw_brute.signal_segments['num segment : 0'],
                             event=all_event['num segment : 0'], name='raw plot',
                             option_csc=['CSC2', 'CSC6', 'CSC7', 'CSC8'])


    plot.plot_waveform_neurone(intervalle=intervalle,
                               neurones_index_val=[neurone.data['time_index_in raw'][data_brute[0].index]],
                               raw_signal=raw_brute.signal_segments['num segment : 0'],
                               event=all_event,
                               name='raw plot', option_csc=['CSC2', 'CSC6', 'CSC7', 'CSC8'])




    print('ifn')