import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure
from matplotlib.axes._axes import Axes
import numpy as np
from numpy.core.multiarray import ndarray
from typing import List, Union, Tuple
from py_NPClab_Package.traitement_labview.Labview_traitment import AnalyseFromLabview
from py_NPClab_Package.traitement_deeplabcut.DeepLabCut_traitment import PreFormDLC

Norme_Coordonnee_Vecteur = Tuple[Series, Series]
Trajectoire = Tuple[ndarray, ndarray, ndarray]


class BasicPlot(object):

    @staticmethod
    def crea_fig() -> Figure:
        """
        Création de la figure
        :return:
        """
        fig = plt.figure(4, figsize=(20, 20))
        return fig

    @staticmethod
    def axis_list(items: List[str], fig1: Figure) -> List[Axes]:
        """
        Création des axes
        :param items:
        :param fig1:-
        :return:
        """
        axis_list: List = []
        if len(items) != 1:
            for ind_axsubplot in range(1, len(items) + 1):
                axis_list.append(fig1.add_subplot(np.ceil(len(items) / 2), 2, ind_axsubplot))
        else:
            axis_list.append(fig1.add_subplot(len(items), 1, 1))
        return axis_list


class SpecifiquePlot(object):

    def plot_traj(self, data: DataFrame, *args):

        fig1 = BasicPlot.crea_fig()

        colors = ['g', 'b', 'r', 'g', 'b', 'r', 'g', 'b', 'r']
        if len(args) > 1:
            if args[1] == 'DLC':
                items_traj: List[str] = args[0]
                axis_list = BasicPlot.axis_list(items_traj, fig1)
                ax: Axes
                for idx, ax in enumerate(axis_list):
                    # ax.plot(self.data[args[idx][0]][171:175], self.data[args[idx][1]][171:175], colors[idx]+'o-')
                    ax.plot(data[args[0][idx]]['x'], data[args[0][idx]]['y'], colors[idx]+'-')
                    ax.set(xlim=(0, 800), ylim=(0, 800))
                    ax.set_xlabel(args[0][idx])

            if args[1] == 'AFL':
                ax: Axes
                axis_list = BasicPlot.axis_list(['trajectoire'], fig1)
                items_traj: List[str] = args[0]
                for idx, ax in enumerate(axis_list):
                    # ax.plot(self.data[args[idx][0]][171:175], self.data[args[idx][1]][171:175], colors[idx]+'o-')
                    ax.plot(data[items_traj[0]], data[items_traj[1]], colors[idx] + '-')
                    ax.set(xlim=(0, 1800), ylim=(0, 1800))
                    ax.set_xlabel(args[1])
        else:
            ax: Axes
            items_traj: List[str] = args[0]
            axis_list = BasicPlot.axis_list(items_traj, fig1)

            for idx, ax in enumerate(axis_list):
                # ax.plot(self.data[args[idx][0]][171:175], self.data[args[idx][1]][171:175], colors[idx]+'o-')
                ax.plot(data[args[idx][0]], data[args[idx][1]], colors[idx] + 'o-')
                ax.set(xlim=(0, 800), ylim=(0, 800))
                ax.set_xlabel(args[0][idx])
        fig1.show()

    def plot_norme_vecteur(self, norme_vecteur: DataFrame, option_event: Union[PreFormDLC, AnalyseFromLabview] = None):
        """
        les events sont automatiquement plot sur le vecteur
        """
        try:
            assert len(norme_vecteur) > 0

            assert isinstance(norme_vecteur, DataFrame)

            fig1 = BasicPlot.crea_fig()
            items_norme_vecteur: List[str] = list(norme_vecteur.keys())
            axis_list = BasicPlot.axis_list(items_norme_vecteur, fig1)
            colors =['g', 'b', 'r','g', 'b', 'r','g', 'b', 'r']
            ax: Axes
            if not isinstance(option_event, AnalyseFromLabview):
                for idx, ax in enumerate(axis_list):
                    ax.plot(np.array(norme_vecteur.loc[:, [items_norme_vecteur[idx]]]), colors[idx])
                    # ax.set(xlim=(000, 700), ylim=(20, 80))
                    ax.set(ylim=(0, 200))
                    ax.set_xlabel(items_norme_vecteur[idx])
            else:
                for idx, ax in enumerate(axis_list):
                    ax.plot(np.array(norme_vecteur.loc[:, [items_norme_vecteur[idx]]]), colors[idx])
                    ax = option_event.plot_event(idx, ax)
                    # for i in option_event.reward_from_txt.index:
                    #     ax.plot(
                    #         [option_event.time_structure.loc[i, 'time_event_x'], option_event.time_structure.loc[i, 'time_event_x']],
                    #         [option_event.time_structure.loc[i, 'time_event_y'], option_event.time_structure.loc[i, 'time_event_y'] - 100], colors[idx] + 'o-')
                    ax.set(xlim=(0, 2000), ylim=(-50, 80))
                    # ax.set(ylim=(norme_vecteur[items_norme_vecteur[idx]].min(), norme_vecteur[items_norme_vecteur[idx]].max()))

                    # ax.set(xlim=(0, 1000), ylim=(norme_vecteur[items_norme_vecteur[idx]].min(), norme_vecteur[items_norme_vecteur[idx]].max()))
                    ax.set_xlabel(items_norme_vecteur[idx])
            fig1.show()
        except TypeError:
            raise TypeError('Ce n"est pas un DataFrame')
        except ValueError:
            raise ValueError('Problème de valeurs d"entrée')

    def plot_acceleration(self, DeltaV: DataFrame):
        fig1 = BasicPlot.crea_fig()
        items_acceleration: List[str] = list(DeltaV.keys())
        axis_list = BasicPlot.axis_list(items_acceleration, fig1)
        colors =['g', 'b', 'r','g', 'b', 'r','g', 'b', 'r']
        ax: Axes
        for idx, ax in enumerate(axis_list):
            ax.plot(np.array(DeltaV.loc[:, [items_acceleration[idx]]]), colors[idx])
            ax.set(ylim=(-50, 50))
            ax.set_xlabel(items_acceleration[idx])
        fig1.show()

    def plot_zone_brute(self, data: DataFrame, items: List[str]):
        fig, ax = plt.subplots(1, figsize=(20, 20))
        ax.set(xlim=(0, 800), ylim=(0, 800))
        colors = ['g', 'b', 'r']

        # ax.plot(data.loc[[0], [items[1]]], data.loc[[0], [items[2]]], 'bo', markersize=12)
        # ax.plot(self.data.loc[[0], [self.items[4]]], self.data.loc[[0], [self.items[5]]], 'bo', markersize=12)
        # ax.plot(self.data.loc[[0], [self.items[7]]], self.data.loc[[0], [self.items[8]]], 'ro', markersize=12)
        # ax.plot(self.data.loc[[0], [self.items[10]]], self.data.loc[[0], [self.items[11]]], 'go', markersize=12)
        # ax.plot(self.data.loc[[0], [self.items[13]]], self.data.loc[[0], [self.items[14]]], 'bo', markersize=12)
        # ax.plot(self.data.loc[[0], [self.items[16]]], self.data.loc[[0], [self.items[17]]], 'bo', markersize=12)
        # ax.plot(self.data.loc[[0], [self.items[19]]], self.data.loc[[0], [self.items[20]]], 'bo', markersize=12)
        # ax.plot(self.data.loc[[0], [self.items[22]]], self.data.loc[[0], [self.items[23]]], 'bo', markersize=12)
        for i in range(0,2000,20):
            x = [np.array(data.loc[[i], [items[7]]])[0][0], np.array(data.loc[[i], [items[10]]])[0][0]]
            y = [np.array(data.loc[[i], [items[8]]])[0][0], np.array(data.loc[[i], [items[11]]])[0][0]]

            ax.plot(x,y)

        fig.show()

    def plot_vecteurs(self, data: DataFrame, items: List[str]):
        fig, ax = plt.subplots(1, figsize=(20, 20))
        ax.set(xlim=(0, 800), ylim=(0, 800))
        colors = ['g', 'b', 'r']

        # ax.plot(data.loc[[0], [items[1]]], data.loc[[0], [items[2]]], 'bo', markersize=12)

        # for i in range(0,2000,20):
        for i in range(0, 2, 1):
            ax.plot(data.loc[[i], [items[7]]], data.loc[[i], [items[8]]], 'yo', markersize=12)

            ABx = [np.array(data.loc[[i], [items[1]]])[0][0], np.array(data.loc[[i], [items[4]]])[0][0]]
            ABy = [np.array(data.loc[[i], [items[2]]])[0][0], np.array(data.loc[[i], [items[5]]])[0][0]]
            ax.plot(ABx,ABy,'b')
            ACx = [np.array(data.loc[[i], [items[1]]])[0][0], np.array(data.loc[[i], [items[7]]])[0][0]]
            ACy = [np.array(data.loc[[i], [items[2]]])[0][0], np.array(data.loc[[i], [items[8]]])[0][0]]
            ax.plot(ACx,ACy,'r')
            BCx = [np.array(data.loc[[i], [items[4]]])[0][0], np.array(data.loc[[i], [items[7]]])[0][0]]
            BCy = [np.array(data.loc[[i], [items[5]]])[0][0], np.array(data.loc[[i], [items[8]]])[0][0]]
            ax.plot(BCx,BCy,'g')

        fig.show()

    def plot_event_around(self, items_norme_vecteur: List[str], around_event: DataFrame):
        """
        les données doivent être préparées
        :param items_norme_vecteur:
        :param around_event:
        :return:
        """
        fig1 = BasicPlot.crea_fig()
        deb = list(range(0, len(around_event), 10))
        fin = list(range(9, len(around_event), 10))
        event_mean = [np.array(around_event.loc[d:f].mean()) for d, f in zip(deb, fin)]
        event_std = [np.array(around_event.loc[d:f].std()) for d, f in zip(deb, fin)]
        axis_list = BasicPlot.axis_list(items_norme_vecteur, fig1)
        colors = ['g', 'b', 'r', 'g', 'b', 'r', 'g', 'b', 'r']
        ax: Axes
        for idx, ax in enumerate(axis_list):
            y = np.array(event_mean[idx])
            x = list(range(-20, 21, 1))
            std = np.array(event_std[idx])
            ax.errorbar(x, y, std, linestyle='-', marker='^')
            ax.plot([0, 0], [0, 20], colors[idx] + 'o-')
            ax.set(ylim=(0, 50))
            ax.set_xlabel(items_norme_vecteur[idx])
        fig1.show()

    def plot_aire(self, aire: DataFrame):
        fig1 = BasicPlot.crea_fig()
        items_triangle = list(aire.keys())
        axis_list = BasicPlot.axis_list(items_triangle, fig1)
        colors = ['g', 'b', 'r', 'g', 'b', 'r', 'g', 'b', 'r']
        ax: Axes
        for idx, ax in enumerate(axis_list):
            ax.plot(aire[items_triangle], colors[idx])
            ax.set(ylim=(0, 300))
            ax.set_xlabel(items_triangle[idx])
        fig1.show()

    def plot_serie(self, serie: Series):
        fig1 = BasicPlot.crea_fig()
        items = ['series']
        axis_list = BasicPlot.axis_list(items, fig1)
        colors = ['g', 'b', 'r', 'g', 'b', 'r', 'g', 'b', 'r']
        ax: Axes
        for idx, ax in enumerate(axis_list):
            ax.plot(serie.loc[:], colors[idx])
            ax.set(ylim=(0, 200))
            # ax.set_xlabel(items_triangle[idx])
        fig1.show()

if __name__ == '__main__':
    from py_NPClab_Package.utilitaire_load import LabviewFilesReward, LabviewFilesTrajectory, LoadData
    from traitement_labview.Labview_traitment import AnalyseFromLabview

    # ------------------------------------- parti import data labview ---------------------------------------------------

    dir_txt_traj = r'D:\Dropbox\python\import_neuralynxv2\data\cplx07 + bsl\fichier traj\*.txt'

    # chargement de la trajectoire contient "x, y, point"
    trajectoire = LoadData.init_data(LabviewFilesTrajectory, dir_txt_traj)
    # chargement des temps en ms "temps" contenu dans le fichier rewards
    reward = LoadData.init_data(LabviewFilesReward, dir_txt_traj)

    # traitement
    data_AFL = AnalyseFromLabview()
    # création d'un dataframe contenant "trajectoire_x, trajectoire_y, reward, rewards, omissions"
    data_tracking_AFL = data_AFL.load_data_from_labview(trajectoire, reward)

    # ------------------------------------- parti traitement de la trajectoire labview -----------------
    from utilitaire_traitement.TrajectoryTraitement import BasicTraitmentTrajectory

    traitment_AFL = BasicTraitmentTrajectory()
    data_traiter_AFL = traitment_AFL.correction(data_AFL, data_AFL.format_correction)

    # ------------------------------------- parti preparation des données pour le plot ---------
    from utilitaire_traitement.PreFormatData import PreFormatData
    data_formater = PreFormatData()

    omission_time = pd.Series(data_AFL.omission, name='omission')

    data_formater.make_event_around(data=traitment_AFL.norme_vecteur, reward=omission_time)

    # ------------------------------------- parti plot ----------------------------------------
    plotcomportement = SpecifiquePlot()
    plotcomportement.plot_event_around(['trajectoire'], data_formater.around_event)
    plotcomportement.plot_norme_vecteur(traitment_AFL.norme_vecteur, data_AFL)
    plotcomportement.plot_traj(data_tracking_AFL, data_AFL.couple_de_points[0])
