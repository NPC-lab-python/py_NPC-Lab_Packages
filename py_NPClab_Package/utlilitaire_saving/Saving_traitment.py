import shelve as sh
import glob
from typing import List, Tuple, Union
import logging
import os
import numpy as np
from numpy.core.multiarray import ndarray
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from collections import defaultdict
from py_NPClab_Package.utilitaire_traitement.Decorateur import mesure


logging.basicConfig(level=logging.DEBUG)


class VerifDossier(object):

    def existe_folder_saving(self, dir_data: str = None, name_folder: str = None, other_path: str = None) -> Tuple[int, str]:
        """
        Cette methode permet de vérifier si le dossier "save" existe et sinon de le créer.

        :param dir_data: correspond au chemin de base du repertoire
        :param name_folder: on peut choisir le nom du dossier à créer par defaut c'est "save"
        :return:
        """
        if isinstance(name_folder, type(None)):
            name_folder = 'save'

        if isinstance(other_path, type(None)) and not isinstance(dir_data, type(None)):
            folder_current = os.getcwd()
            logging.debug(f'Dossier courant actuel {folder_current}')
            os.chdir(dir_data)
            new_folder_current = os.getcwd()
        else:
            logging.debug(f'Dossier courant actuel {other_path}')
            os.chdir(other_path)
            new_folder_current = os.getcwd()

        name = f'\\{name_folder}'

        if os.path.isdir(new_folder_current + name):
            logging.debug(f'Dossier {name_folder} existant')
            path = new_folder_current + name
            return 0, path
        else:
            logging.debug(f'Création du repertoire {name_folder}')
            os.mkdir(name_folder)
            path = new_folder_current + name

            return 1, path


class SaveSerialisation(VerifDossier):
    """
        "name" correspond au nom du fichier
        "path" correspond au chemin
    """
    def load_data_serializer(self, path: str, name: str):
        """
        Cette méthode permet de charger directement le fichier voulu
        quand on est sure qu'il existe
        :param path: chemin comprend le dossier "save"
        :param name: nom du fichier sans "l'extension"
        :return:
        """
        data = self._chargement_fichier_data_(path, name)
        return data

    def _chargement_fichier_data_(self, path: str, name: str):

        with sh.open(path + '\\' + name) as data:
            data_extract = data[name]
        logging.debug(f'type du fichier loader {type(data_extract)}, name : {name}, path : {path}')
        return data_extract

    def _set_conf_(self, name: str, dir_save_conf: str, data: object) -> object:
        """
        Cette méthode va créer un fichier sérialiser de data

        :param name: correspond au nom qui sera donné au fichier
        :param dir_data:
        :param data:
        :return:
        """
        dir = f'{dir_save_conf}\{name}'
        save_conf = sh.open(dir)
        save_conf[dir.split('\\')[-1]] = data
        save_conf.close()
        logging.debug(f'Création du fichier {name} dans le folder : {dir_save_conf}')

    def _get_conf_(self, name: str, name_folder: str = None, dir_save_conf: str = None, other_path: str = None) -> Tuple[int, str]:
        """
        C'est la première méthode appelé.
        elle permet de vérifier la présence du fichier et d'oriente
        la suite.
        si absent, il sera créé par "_set_conf_"
        si présent, il sera chargé par "_chargement_fichier_data_"

        :param dir_save_conf:
        :param name:
        :param name_folder:
        :return: retourne un tuple contenant en [0] un "int" pour absent ou present
        et en [1] le chemin
        """
        logging.debug(f'Vérification de l"existance du fichier {name}')

        if isinstance(other_path, type(None)) and not isinstance(dir_save_conf, type(None)):
            if isinstance(name_folder, type(None)):
                dir_data = self.existe_folder_saving(dir_save_conf)
            else:
                dir_data = self.existe_folder_saving(dir_data=dir_save_conf, name_folder=name_folder)

            path_save = f'{dir_data[1]}\\*.dat'
            path = [i for i in glob.glob(path_save) if i.find(f'.dat')]
            e = [i for i in path if i.find(name) != -1]
            if len(e) == 0:
                logging.debug(f'le fichier {name} existe pas ou est introuvable')
                return 1, dir_data[1]
            else:
                logging.debug(f'le fichier {name} existe')
                return 0, dir_data[1]

        if not isinstance(other_path, type(None)) and isinstance(dir_save_conf, type(None)):
            if isinstance(name_folder, type(None)):
                dir_data = self.existe_folder_saving(other_path)
            else:
                dir_data = self.existe_folder_saving(other_path=other_path, name_folder=name_folder)

            path_save = f'{dir_data[1]}\\*.dat'
            path = [i for i in glob.glob(path_save) if i.find(f'.dat')]
            e = [i for i in path if i.find(name) != -1]
            if len(e) == 0:
                return 1, dir_data[1]
            else:
                return 0, dir_data[1]


class SavingMethodes(object):

    @staticmethod
    def save_data_text(data: ndarray, name: str, path: str = None, option: str = None):
        extention: str = '.txt'
        final_path = path + name + extention
        np.savetxt(fname=final_path, X=data, fmt='%1.4e')

    def _save_figure_(self, fig: plt, name: str, save: bool = True, ext: str = '.eps', option: str = None):
        if type(option) is str:
            names: str = name + option + ext
        else:
            names = name + ext

        if not save:
            pass
        else:
            fig.savefig(names, format='eps')
        fig.show()

    @mesure
    def preformat_output(self, data: defaultdict) -> DataFrame:
        item_columns = []
        for idx, val in enumerate(data.keys()):
            for i in range(3):
                item_columns.append(val)
        item_coord = []
        for idx, val in enumerate(data.keys()):
            item_coord.extend(data[val].keys())

        coord = ['coord']
        coord.extend(list(range(len(data[[i for i in data.keys()][0]]))))
        output_data = pd.DataFrame(columns=item_columns, index=coord)
        for idx, val in enumerate(data.keys()):
            output_data[val] = data[val]
        output_data.loc['coord'] = item_coord
        output_data.loc['coord'] = item_coord

        return output_data

        # pd.concat([data['centre'], data['nose']], axis='columns')


    def save_csv(self, data: Union[defaultdict, DataFrame], path: str):
        output_data = self.preformat_output(data=data)
        output_data.to_csv(path_or_buf=path, sep=',', header=True)
