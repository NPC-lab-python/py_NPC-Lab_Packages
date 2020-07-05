import sys
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from numpy.core.multiarray import ndarray
from typing import List, Union, Tuple
import glob
from collections import defaultdict
import shelve as sh
import logging

from py_NPClab_Package.utilitaire_traitement.Decorateur import mesure
from py_NPClab_Package.utilitaire_neuralynx.Load_neuralynx import RawSignals

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

Norme_Coordonnee_Vecteur = Tuple[Series, Series]
Trajectoire = Tuple[ndarray, ndarray, ndarray]

from abc import ABC, abstractmethod, abstractclassmethod


class ModelImport(ABC):

    @abstractmethod
    def _init_path_(self, dir: str) -> List[str]:
        pass

    @abstractmethod
    def _load_txt_(self, path: str) -> Union[ndarray, Trajectoire, Series]:
        pass

    @abstractmethod
    def _manage_files_(self):
        pass

    @abstractmethod
    def _final_output_(self) -> defaultdict:
        pass

    @abstractclassmethod
    class InitLoader(object):
        @abstractmethod
        def load(self, dir: str):
            pass


class LoadData(object):
    fabrique = {}

    @staticmethod
    def init_data(cls, *args):
        id = cls.__name__
        if not id in LoadData.fabrique:
            LoadData.fabrique[id] = cls.InitLoader()
        return LoadData.fabrique[id].load(*args)


class NeuralynxFilesSpike(ModelImport):

    def __init__(self, dir: str):
        self.dir: str = dir
        self.path: List[str] = self._init_path_(dir)
        self.name_file: ndarray = None
        self.data: defaultdict = self._manage_files_(self.path)

    def _init_path_(self, dir: str) -> List[str]:
        path = [i for i in glob.glob(dir) if i.split('\\')[-1].find('SpikeFile') == 0]
        try:
            assert len(path) > 0
        except AssertionError as e:
            logging.debug(f'Chemin du fichier spike est erroné ou le nom ne contient pas "SpikeFile" ligne: 73')
            sys.exit()
        return path

    def _manage_files_(self, path: List[str]) -> defaultdict:
        data_file = self._final_output_()
        # data_file: List[Series] = []
        name_files: ndarray = np.array([], dtype=str)
        for idx, file in enumerate(path):
            name_files = np.hstack((name_files, file.split('\\')[-1]))
            data_file[idx] = pd.DataFrame(self._load_txt_(file), columns=[file.split('\\')[-1]])
            self.name_file = name_files
        return data_file

    def _final_output_(self) -> defaultdict:
        # items_segment = self._liste_segment_(event_brute)
        final_output = defaultdict(DataFrame)
        return final_output

    def _load_txt_(self, path: str) -> Series:
        spike_raw: ndarray = np.array([], dtype=int)
        with open(path, 'r') as file:
            for line in file:
                spike_raw = np.hstack((spike_raw, int(line.split('\n')[0])))
        event_time: Series = pd.Series(spike_raw)
        return event_time

    class InitLoader(object):

        def load(self, dir: str):
            return NeuralynxFilesSpike(dir)


class LabviewFilesTrajectory(object):

    def __init__(self, dir):
        self.dir: str = dir
        self.path: List[str] = self._init_path_(dir)
        self.name_file = self.path[0].split('\\')[-1]
        self.data: Trajectoire = self._load_txt_(self.path)

    def _init_path_(self, dir: str) -> List[str]:
        path_traj = [i for i in glob.glob(dir) if i.find('traj.txt') != -1]
        try:
            assert len(path_traj) > 0
        except AssertionError as e:
            logging.debug(f"Chemin du fichier traj erroné (Basic_Load : ligne119)")
            sys.exit()

        return path_traj

    def _load_txt_(self,path_traj) -> Trajectoire:

        trajectoire_x: ndarray = np.array([], dtype=float)
        trajectoire_y: ndarray = np.array([], dtype=float)
        reward_raw: ndarray = np.array([], dtype=int)
        print(f'path : {path_traj}')
        with open(path_traj[0], 'r') as file:
            for line in file:
                trajectoire_x = np.hstack((trajectoire_x, float(line.split('\t')[0].replace(',','.'))))
                trajectoire_y = np.hstack((trajectoire_y, float(line.split('\t')[1].replace(',','.'))))
                if float(line.split('\t')[2].split(',')[0]) == -1:
                    reward_raw = np.hstack((reward_raw, int(float(line.split('\t')[2].split(',')[0])**10*10)))
                else:
                    reward_raw = np.hstack((reward_raw, int(float(line.split('\t')[2].split(',')[0]))))

        return trajectoire_x, trajectoire_y, reward_raw

    class InitLoader(object):
        def load(self,dir):
            return LabviewFilesTrajectory(dir)


class LabviewFilesReward(object):

    def __init__(self, dir: str):
        self.dir: str = dir
        self.path: List[str] = self._init_path_(dir)
        self.name_file = self.path[0].split('\\')[-1]
        self.data: Series = self._load_txt_(self.path)

    def _init_path_(self, dir: str) -> List[str]:
        path = [i for i in glob.glob(dir) if i.find('reward.txt') != -1]
        try:
            assert len(path) > 0
        except AssertionError as e:
            logging.debug(f'Chemin du fichier reward erroné ou '
                          f' Le fichier doit ce terminer par "*reward.txt')
            sys.exit()
        return path

    def _load_txt_(self, path: List[str]) -> Series:
        reward_raw: ndarray = np.array([], dtype=int)
        with open(path[0], 'r') as file:
            for line in file:
                reward_raw = np.hstack((reward_raw, int(line.split('\t')[0])))
        event_time: Series = pd.Series(reward_raw)
        return event_time

    class InitLoader(object):
        def load(self, dir: str):
            return LabviewFilesReward(dir)


class NeuroneFilesSerialiser(object):

    def __init__(self, dir: str, name: str):
        self.dir: str = dir
        self.path: List[str] = self._init_path_(dir, name)
        self.name_file = self.path[0].split('\\')[-1]
        self.data: Series = self._load_serialiser_file_(self.path, name)

    def _init_path_(self, dir: str, name: str) -> List[str]:


        path_save = f'{dir}\\*.dat'
        path = [i for i in glob.glob(path_save) if i.find(f'.dat')]
        try:
            assert len(path) > 0
        except AssertionError as e:
            logging.debug(f'Il y a un problème avec le chemin du fichier '
                          f'ou le fichier "neurone" est introuvable ')
            sys.exit()
        e = [i for i in path if i.find(name) != -1]
        return e

    def _load_serialiser_file_(self, path: List[str], name: str) -> Series:
        with sh.open(path[0][:-4]) as data:
            data_extract = data[name]
            data_extract = pd.Series(data_extract)
        return data_extract


    class InitLoader(object):
        def load(self, dir: str, name: str):
            return NeuroneFilesSerialiser(dir, name)

class EventFilesSerialiser(object):

    def __init__(self, dir: str, name: str):
        self.dir: str = dir
        self.path: List[str] = self._init_path_(dir, name)
        self.name_file = self.path[0].split('\\')[-1]
        self.data: defaultdict = self._manage_files_(name)

    def _init_path_(self, dir: str, name: str) -> List[str]:
        path_save = f'{dir}\\*.dat'
        path = [i for i in glob.glob(path_save) if i.find(f'.dat')]
        try:
            assert len(path) > 0
        except AssertionError as e:
            logging.debug(f'Il y a un problème avec le chemin du fichier '
                          f'ou le fichier "{name}" est introuvable ')
            sys.exit()
        e = [i for i in path if i.find(name) != -1]
        return e
    def _manage_files_(self,name):
        data: dict = self._load_serialiser_file_(self.path, name)
        data_file: defaultdict = data['all_event']
        return data_file

    def _load_serialiser_file_(self, path: List[str], name: str) -> dict:
        with sh.open(path[0][:-4]) as data:
            data_extract = data[name]
            # data_extract = pd.Series(data_extract)
        return data_extract


    class InitLoader(object):
        def load(self, dir: str, name: str):
            return EventFilesSerialiser(dir, name)


class SegmentBTFilesSerialiser(object):
    """
    "segment_infos" est constitué de series de dictionnaire
    chaque series est nomé et correspond à un segment
    exemple : si la ssesion comporte deux expérimentations ("base_line et compléxité"),
            segment0 correspond à la base_line
            segment1 correspond à la compléxité

    """

    def __init__(self, dir: str, name: str, num_segment: int):
        self.dir: str = dir
        self.path: List[str] = self._init_path_(dir, name)
        self.name_file = self.path[0].split('\\')[-1]
        self.data: ndarray = self._manage_files_(num_segment=num_segment, name=name)

    def _init_path_(self, dir: str, name: str) -> List[str]:
        path_save = f'{dir}\\*.dat'
        path = [i for i in glob.glob(path_save) if i.find(f'.dat')]
        try:
            assert len(path) > 0
        except AssertionError as e:
            logging.debug(f'Il y a un problème avec le chemin du fichier '
                          f'ou le fichier {name} est introuvable ')
            sys.exit()
        e = [i for i in path if i.find(name) != -1]
        return e

    def _manage_files_(self, num_segment: int, name: str) -> ndarray:
        """
        Cetteméthode permet de reloader le fichier "segment_infos" contenu dans le dossier
        "save" créé précedent. Pour ne prendre que la "base_time".

        """
        data: defaultdict = self._load_serialiser_file_(self.path, name)
        items_segment = {}
        for i in data.keys():
            cle = i.split(': ')[1]
            items_segment[cle] = i
        data_file: ndarray = data[items_segment[str(num_segment)]]['base_time']
        return data_file

    @mesure
    def _load_serialiser_file_(self, path: List[str], name: str) -> defaultdict:
        with sh.open(path[0][:-4]) as data:
            data_extract = data[name]
        return data_extract

    class InitLoader(object):
        def load(self, dir: str, name: str, num_segment: int):
            return SegmentBTFilesSerialiser(dir, name, num_segment)


class SegmentFilesSerialiser(object):
    """
    "segment_infos" est constitué de series de dictionnaire
    chaque series est nomé et correspond à un segment
    exemple : si la ssesion comporte deux expérimentations ("base_line et compléxité"),
            segment0 correspond à la base_line
            segment1 correspond à la compléxité

    """

    def __init__(self, dir: str, name: str):
        self.dir: str = dir
        self.path: List[str] = self._init_path_(dir, name)
        self.name_file = self.path[0].split('\\')[-1]
        self.data: defaultdict = self._manage_files_(name=name)

    def _init_path_(self, dir: str, name: str) -> List[str]:
        path_save = f'{dir}\\*.dat'
        path = [i for i in glob.glob(path_save) if i.find(f'.dat')]
        try:
            assert len(path) > 0
        except AssertionError as e:
            logging.debug(f'Il y a un problème avec le chemin du fichier '
                          f'ou le fichier {name} est introuvable ')
            sys.exit()
        e = [i for i in path if i.find(name) != -1]
        return e

    def _manage_files_(self, name: str) -> defaultdict:
        """
        Cette méthode permet de reloader le fichier "segment_infos" contenu dans le dossier
        "save" créé précedent.

        """
        data_file: defaultdict = self._load_serialiser_file_(self.path, name)
        return data_file

    @mesure
    def _load_serialiser_file_(self, path: List[str], name: str) -> defaultdict:
        with sh.open(path[0][:-4]) as data:
            data_extract = data[name]
        return data_extract


    class InitLoader(object):
        def load(self, dir: str, name: str):
            return SegmentFilesSerialiser(dir, name)


class DeepLabCutFileImport(ModelImport):
    """
    Cette class permet d'importer simplement le csv de DeepLabCut.
    Un seul ficier à la fois .
    A voir sil il faut cela !!
    """
    def __init__(self, dir: str, name: str):
        self.dir: str = dir
        self.path: List[str] = self._init_path_(dir, name)
        self.name_file = self.path[0].split('\\')[-1]
        self.data: DataFrame = self._manage_files_()

    def _init_path_(self, dir: str, name: str) -> List[str]:
        path_save = f'{dir}\\*.csv'
        path = [i for i in glob.glob(path_save) if i.find(f'.csv')]
        try:
            assert len(path) > 0
        except AssertionError as e:
            logging.debug(f'Il y a un problème avec le chemin du fichier '
                          f'ou le fichier {name} est introuvable')
            sys.exit()
        e = [i for i in path if i.find(name) != -1]
        return e

    def _load_txt_(self, path: str) -> DataFrame:
        data_brute: DataFrame = pd.read_csv(path)
        return data_brute

    def _manage_files_(self):
        data: DataFrame = self._load_txt_(self.path[0])
        return data

    def _final_output_(self) -> defaultdict:
        pass

    class InitLoader(object):
        def load(self, dir: str, name: str):
            return DeepLabCutFileImport(dir, name)


class ImportNeuralynx(RawSignals):

    def __init__(self, csc_dir: str, option: str):
        super().__init__(csc_dir)
        self.load(option=option)

    class InitLoader(object):
        def load(self, csc_dir: str, option: str):
            return ImportNeuralynx(csc_dir, option)


if __name__ == '__main__':
    """
    Pour loader un neurone : 
    dir_save = r'D:\Dropbox\python\import_neuralynxv2\data\cplx07 + bsl\save'
    name = 'segment1_neurone0'
    neurone = LoadData.init_data(NeuroneFilesSerialiser, dir_save, name)

    """

    # dir_txt = r'/data/cplx07 + bsl/fichier traj/shenron1_cplx_07_filter1_08042020-1527_reward.txt'
    # dir_csv = r'D:\Dropbox\python\import_neuralynxv2\data\trajectoire_tarek_deeplapcut\Chichi_3_cplx_04bis_500mV_03042020-1627DLC_mobnet_100_FiberApr1shuffle1_160000.csv'
    # dir_txt_traj = r'D:\Dropbox\python\import_neuralynxv2\data\cplx07 + bsl\fichier traj\shenron1_cplx_07_filter1_08042020-1527traj.txt'

    # dir_txt_traj = r'D:\Dropbox\python\import_neuralynxv2\data\cplx07 + bsl\fichier traj\*.txt'
    # dir_neuralynx = r'D:\Dropbox\python\import_neuralynxv2\data\cplx07 + bsl'
    # dir_spikefile = r'D:\Dropbox\python\import_neuralynxv2\data\cplx07 + bsl\clustering\*.txt'
    #
    # dir_save = r'D:\Dropbox\python\import_neuralynxv2\data\cplx07 + bsl\save'
    # name = 'segment1_neurone0'
    # neurone = LoadData.init_data(NeuroneFilesSerialiser, dir_save, name)

    # trajectoire = LoadData.init_data(LabviewFilesTrajectory, dir_txt_traj)
    # reward = LoadData.init_data(LabviewFilesReward, dir_txt_traj)
    # Event_brute = LoadData.init_data(ImportNeuralynx, dir_neuralynx, 'event')
    # dir_DLC = r'D:\Dropbox\python\import_neuralynxv2\data\trajectoire_tarek_deeplapcut\DLC_data'
    #
    # DLC_brute = LoadData.init_data(DeepLabCutFileImport, dir_DLC, 'Chichi_3_cplx_04bis_500mV_03042020-1627DLC_resnet50_testgpuMay2shuffle1_50000.csv')

    dir_save = r'/data/cplx07 + bsl/save'
    name = 'segment_infos'
    segment_infos_loaded = LoadData.init_data(SegmentFilesSerialiser, dir_save, name,1)

    print('fin')

