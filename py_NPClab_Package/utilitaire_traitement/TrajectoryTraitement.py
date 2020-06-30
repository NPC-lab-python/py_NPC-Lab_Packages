# -*-coding:utf-8 -*-
import pandas as pd
from pandas import DataFrame, Series, RangeIndex
import numpy as np
from scipy.interpolate import interp1d
from numpy.core.multiarray import ndarray
from typing import List, Union, Tuple
from collections import defaultdict

from py_NPClab_Package.utilitaire_traitement.Decorateur import mesure
from py_NPClab_Package.traitement_deeplabcut.DeepLabCut_traitment import PreFormDLC

import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

Norme_Coordonnee_Vecteur = Tuple[Series, Series]
Trajectoire = Tuple[ndarray, ndarray, ndarray]


class BasicCorrectionTrajectory(object):
    """
    cette class permet de calcul les normes des vecteurs.
    si on rentre la trajectoire (labview) on aura la vitesse instannée.
    la trajectoire est aussi corrigée. Dans le cas ou les couples de points sont le reflet de frame.
    sinon cela représente la longueur des vecteurs
    Pour utiliser cette class :

    """
    def __init__(self):
        self.norme_vecteur: DataFrame = pd.DataFrame()
        self.DeltaV: DataFrame = pd.DataFrame()
        self.norme_vecteur_entre_point: DataFrame = pd.DataFrame()
        self.coordonnee_vecteur_entre_point: DataFrame = pd.DataFrame()
        self.around_event: DataFrame = pd.DataFrame()


    def _calcul_norme_(self, x1: Series, x2: Series, y1: Series, y2: Series) -> Series:
        """
        calcul de la norme du vecteur ou de la longueur du vecteur
        :param x1:
        :param x2:
        :param y1:
        :param y2:
        :return:
        """
        norme_vecteur: Series = pd.Series(np.sqrt((x2-x1)**2+(y2-y1)**2), name='norme_du_vecteur')
        return norme_vecteur

    @mesure
    def _coordonnee_vecteur_(self, x1: Series, x2: Series, y1: Series, y2: Series) -> Series:
        """


        :param x1:
        :param x2:
        :param y1:
        :param y2:
        :return:
        """
        coordonnee_vecteur: Series = pd.Series([(x2 - x1), (y2 - y1)], name='coordonnee_vecteur', dtype=float)
        return coordonnee_vecteur

    @mesure
    def _norme_vecteurv2_(self, data: DataFrame, name_pointsA: str, index0: RangeIndex = None, index1: RangeIndex = None) -> Series:
        """
        calcul de la longueur du vecteur entre deux frame pour le même point
        donc de la norme du vecteur ||name_pointsA[0]name_pointsA[1]||
        ou entre deux points dans la même frame

        :param data:
        :param items:
        :return:
        """
        if not isinstance(index0, RangeIndex) and not isinstance(index1, RangeIndex):
            print('Il manque une entrée !!!')
        else:
            rr = ['x', 'y']
            norme_AB = pd.DataFrame(dtype=float)
            idx = 0
            for _0, _1 in zip(rr, rr):
                idx +=1
                norme_AB[_0[-1]+str(idx)] = data[_0][index0]
                tmp = data[_1][index1]
                tmp.reset_index(drop=True, inplace=True)
                norme_AB[_1[-1]+str(idx+1)] = tmp
                idx = 0
            items = norme_AB.keys()

            norme_vecteur = self._calcul_norme_(x1=norme_AB[items[0]], x2=norme_AB[items[1]], y1=norme_AB[items[2]], y2=norme_AB[items[3]])
            self.norme_vecteur[name_pointsA] = norme_vecteur
            return norme_vecteur

    @mesure
    def _norme_vecteur_smooth_(self, nb_point: int = 5):
        """
        Cette méthode permet de moyenné sur nb_point points
        """

        ee = []
        a = np.array(self.norme_vecteur)
        for i in range(len(a) - nb_point):
            ee.append(a[i:i + nb_point].mean())
        return pd.DataFrame(pd.Series(ee), columns=['norme_smooth'])

    @mesure
    def _norme_vecteur_entre_point_(self, data: DataFrame, name_pointsA: str, name_pointsB: str) -> Union[
        Series, Norme_Coordonnee_Vecteur]:
        """
        calcul de la longueur du vecteur entre deux frame pour le même point
        donc de la norme du vecteur ||name_pointsA[0]name_pointsA[1]||
        ou entre deux points dans la même frame

        :param data:
        :param items:
        :return:
        """
        norme_AB = pd.DataFrame()
        for idx, val in enumerate([name_pointsA, name_pointsB]):
            norme_AB['x' + str(idx + 1)] = pd.Series(data[val]['x'], data[val]['x'].index)
            norme_AB['y' + str(idx + 1)] = pd.Series(data[val]['y'], data[val]['y'].index)

        items = norme_AB.keys()

        coordonnee_vecteur_entre_point = self._coordonnee_vecteur_(x1=norme_AB[items[0]], x2=norme_AB[items[2]],
                                                                   y1=norme_AB[items[1]], y2=norme_AB[items[3]])
        norme_vecteur_entre_point = self._calcul_norme_(x1=norme_AB[items[0]], x2=norme_AB[items[2]],
                                                        y1=norme_AB[items[1]], y2=norme_AB[items[3]])

        self.norme_vecteur_entre_point[name_pointsA + ' ' + name_pointsB] = norme_vecteur_entre_point
        self.coordonnee_vecteur_entre_point[name_pointsA + ' ' + name_pointsB] = coordonnee_vecteur_entre_point

        return norme_vecteur_entre_point, coordonnee_vecteur_entre_point


    def _seuillage_norme_vecteur_(self, data: DataFrame, seuil: int, norme_vecteur: Series):
        """
        Si des points aberrants sont détecter on lance la correction et on enregistre
        :param data:
        :param seuil:
        :param norme_vecteur:
        :param items:
        :return:
        """
        point_aberrant = norme_vecteur[norme_vecteur > seuil]
        if len(point_aberrant) > 0:
            start, stop = self._extract_start_stop_(point_aberrant)

            _cahe_data_work_ = self._interpolation_val_aberrant_(data, start, stop)
            return _cahe_data_work_
        else:
            return 0

    def _extract_start_stop_(self, point_aberrant: Series) -> Tuple[List[int], List[int]]:
            n: int = len(point_aberrant)
            init: int = 0
            start: List[int] = []
            stop: List[int] = []
            self._start_stop_recur_(n, init, point_aberrant, start, stop)
            return start, stop

    def _start_stop_recur_(self, n: int, init: int, point_aberrant: Series, start: list, stop: list):
        """
        fonction recurante qui le début et la fin de chaque séquence de point abérrant
        :param n:
        :param init:
        :param point_aberrant:
        :param start:
        :param stop:
        :return:
        """
        if n == 0 or init >= len(point_aberrant)-1:
            print(n)
            return start, stop
        else:
            idx = 0
            on = []
            while point_aberrant.index[init] + idx in point_aberrant.index:
                idx += 1
                on.append(point_aberrant.index[init]+idx)
            if len(start) < 1:
                if point_aberrant.index[init] - 1 < 0:
                    start.append(point_aberrant.index[init])
                else:
                    start.append(point_aberrant.index[init] - 1)
                stop.append(point_aberrant.index[init] + idx + 1)
                init = init + idx
            else:
                if point_aberrant.index[init] - 1 <= stop[-1]:
                    stop[-1] = point_aberrant.index[init] + idx + 1
                    init = init + idx
                else:
                    start.append(point_aberrant.index[init] - 1)
                    stop.append(point_aberrant.index[init] + idx + 1)
                    init = init + idx
            return self._start_stop_recur_(n - 1, init, point_aberrant, start, stop)

    def _interpol_val_recur_(self, data: DataFrame, x: ndarray, y: ndarray, start: list, stop: list) -> tuple:
        if len(start) == 0:
            return x, y
        else:
            _x = np.array(data.loc[(range(start[0], stop[0])), ['x']])
            _x = np.array([i[0] for i in _x])
            longueur_x = len(_x)

            _y = np.array(data.loc[(range(start[0], stop[0])), ['y']])
            _y = np.array([i[0] for i in _y])

            f = interp1d(_x, _y)
            _x_interm = np.linspace(_x[0], _x[-1], num=longueur_x, endpoint=True)
            _y_interm = f(_x_interm)
            x = np.hstack((x, _x_interm))
            y = np.hstack((y, _y_interm))
            return self._interpol_val_recur_(data, x, y, start[1:], stop[1:])

    def _interpolation_val_aberrant_(self, data: DataFrame, start: list, stop: list) -> DataFrame:
        """
        on remplace les valeurs corrigées directement dans la series correspondant aux points (x,y) du couple entrée (traj).

        :param data:
        :param start:
        :param stop:
        :return:
        """
        x = np.array([], dtype=float)
        y = np.array([], dtype=float)
        x, y = self._interpol_val_recur_(data, x, y, start, stop)
        index = []
        [index.extend(list(range(start[i], stop[i]))) for i in range(len(start))]
        yvals = pd.Series(y, index, float)
        xvals = pd.Series(x, index, float)
        _cahe_work_data_ = data
        _cahe_work_data_.loc[xvals.index, ['x']] = xvals
        _cahe_work_data_.loc[yvals.index, ['y']] = yvals
        return _cahe_work_data_

    def _calcul_acceleration_(self, norme_vecteur: DataFrame, items_norme_vecteur: str) -> Series:
        """
        instatanée
        vitesse = norme_vecteur
        vitesse = V
        temps = T
        acceleration = delta V / delta T. delta V = V final - V initial, delta T = T final - T initial
        delta T = 1 donc acceleration = V final - V initial

        :return:
        """
        t = list(range(1, len(norme_vecteur.loc[:, [items_norme_vecteur]]), 1))
        e = list(range(0, len(norme_vecteur.loc[:, [items_norme_vecteur]]) - 1, 1))

        _Vinitial = np.array([i[0] for i in np.array(norme_vecteur.loc[e, [items_norme_vecteur]])])
        Vinitial = pd.Series(_Vinitial, list(range(0, len(norme_vecteur.loc[:, [items_norme_vecteur]])-1)))
        _Vfinal = np.array([i[0] for i in np.array(norme_vecteur.loc[t, [items_norme_vecteur]])])
        Vfinal = pd.Series(_Vfinal, list(range(0, len(norme_vecteur.loc[:, [items_norme_vecteur]])-1)))

        DeltaV: Series = Vfinal-Vinitial
        return DeltaV

    def acceleration_instant(self):
        items_norme_vecteur: List[str] = list(self.norme_vecteur.keys())
        for i in range(len(items_norme_vecteur)):
            DeltaV = self._calcul_acceleration_(self.norme_vecteur, items_norme_vecteur[i])
            self.DeltaV[items_norme_vecteur[i]] = DeltaV
    @mesure
    def _acceleration_instant_smooth_(self, nb_point: int = 5):
        """
        Cette méthode permet de moyenné sur nb_point points
        """

        ee = []
        a = np.array(self.DeltaV)
        for i in range(len(a) - nb_point):
            ee.append(a[i:i + nb_point].mean())
        return pd.DataFrame(pd.Series(ee), columns=['acceleration_smooth'])

    @mesure
    def trajectoire_brute(self, data: DataFrame, name_pointsA: str):
        """

        :param data: comprenant une series pour "x" et pour "y"
        :param name_pointsA: correspond à un couple de point de coordonnées ['earr x', 'earr y']
        ou ['trajectoire'] pour les fichier venant de labview
        :return:
        """
        index1 = pd.RangeIndex(1, len(data), 1)
        index0 = pd.RangeIndex(0, len(data) - 1, 1)
        # step 1
        norme_vecteur = self._norme_vecteurv2_(data=data, name_pointsA=name_pointsA, index0=index0, index1=index1)
        self.norme_vecteur_smooth = self._norme_vecteur_smooth_()
        return data
        #TODO a réécrire en  numpy

    @mesure
    def correction_trajectoire(self, data: DataFrame, name_pointsA: str):
        """

        :param data: comprenant une series pour "x" et pour "y"
        :param name_pointsA: correspond à un couple de point de coordonnées ['earr x', 'earr y']
        :return:
        """
        index1 = pd.RangeIndex(1, len(data), 1)
        index0 = pd.RangeIndex(0, len(data) - 1, 1)
        # step 1
        norme_vecteur = self._norme_vecteurv2_(data=data, name_pointsA=name_pointsA, index0=index0, index1=index1)
        self.norme_vecteur_smooth = self._norme_vecteur_smooth_()
        self.acceleration = self.acceleration_instant()
        self.acceleration_smooth = self._acceleration_instant_smooth_()

        seuil: int = 40
        # step 2
        new_data = data.copy()
        _cache_data_work_ = self._seuillage_norme_vecteur_(data, seuil, norme_vecteur)
        if isinstance(_cache_data_work_, DataFrame):
            new_data = _cache_data_work_
        # step 3
        norme_vecteur = self._norme_vecteurv2_(data=new_data, name_pointsA=name_pointsA, index0=index0, index1=index1)
        return new_data
        #TODO a réécrire en  numpy

    def norme_vecteur_entre_points(self, data: DataFrame, name_pointsA: str, name_pointsB: str) -> Union[DataFrame, Norme_Coordonnee_Vecteur]:
        """
        calcul de la longueur du vecteur entre deux points  donc de la norme du vecteur ||itemsAitemsB|| pour chaque frame

        :param name_pointsB:
        :param name_pointsA:
        :param data:
        :return:
        """

        norme_vecteur_entre_point = self._norme_vecteur_entre_point_(data=data, name_pointsA=name_pointsA, name_pointsB=name_pointsB)
        nvep = norme_vecteur_entre_point[0]
        return nvep


class BasicTraitmentTrajectory(BasicCorrectionTrajectory):

    def __init__(self):
        super().__init__()

    def data_brute(self, data_struct: PreFormDLC, data_traj: defaultdict):
        """
        :param data_struct: correspond à "data_AFL" issu de "AnalyseFromLabview"
        :param data_traj: correspond à "data_AFL.format_correction"
        :return:
        """
        for i, name in enumerate(data_struct.items):
            logging.debug((f' trajecroire brute : {name}'))
            data_traj[name] = self.trajectoire_brute(data_traj[name], name_pointsA=name)
        return data_traj

    def correction(self, data_struct: PreFormDLC, data_traj: DataFrame):
        """

        :param data_struct: correspond à "data_AFL" issu de "AnalyseFromLabview"
        :param data_traj:
        :return:
        """
        for i, name in enumerate(data_struct.items):
            logging.debug((f' trajecroire corrigé : {name}'))
            data_traj[name] = self.correction_trajectoire(data_traj[name], name_pointsA=name)
        return data_traj

    def make_segment(self, data_traj: DataFrame, pointA: List, pointB: List, pointC: List) -> None:
        # pointA = data_DLC.couple_de_points[0]
        # pointB = data_DLC.couple_de_points[1]
        # pointC = data_DLC.couple_de_points[2]
        self.norme_vecteur_entre_points(data=data_tracking_DLC, name_pointsA='hang', name_pointsB='hand')

        # triangle_ABC = list(combi([pointA, pointB, pointC], 2))
        # for couple in triangle_ABC:
        #     print(couple[0], couple[1])
        #
        #     self.norme_vecteur_entre_points(data=data_traj, couple_pointsA=couple[0], couple_pointsB=couple[1])

if __name__ == "__main__":
    # dir_txt_traj = r'D:\Dropbox\python\import_neuralynxv2\data\cplx07 + bsl\fichier traj\*.txt'
    #
    # # chargement de la trajectoire contient "x, y, point"
    # trajectoire = LoadData.init_data(LabviewFilesTrajectory, dir_txt_traj)
    # # chargement des temps en ms "temps" contenu dans le fichier rewards
    # reward = LoadData.init_data(LabviewFilesReward, dir_txt_traj)
    #
    # # traitement
    # data_AFL = AnalyseFromLabview()
    # # création d'un dataframe contenant "trajectoire_x, trajectoire_y, reward, rewards, omissions"
    # data_tracking_AFL = data_AFL.load_data_from_labview(trajectoire, reward)

# --------------------------- traitement DLC ---------------------------------------------
    from py_NPClab_Package.traitement_deeplabcut.DeepLabCut_traitment import PreFormDLC
    from py_NPClab_Package.utilitaire_load import LoadData, DeepLabCutFileImport
    from utlilitaire_saving.Saving_traitment import SavingMethodes
    dir_DLC = r'/data/trajectoire_tarek_deeplapcut/DLC_data'
    DLC_brute = LoadData.init_data(DeepLabCutFileImport, dir_DLC,
                                   'Chichi_3_cplx_04bis_500mV_03042020-1627DLC_resnet50_testgpuMay2shuffle1_50000.csv')

    data_DLC = PreFormDLC()
    data_tracking_DLC = data_DLC.load_data_from_DLC(data_brute=DLC_brute.data)

    traitment_DLC = BasicTraitmentTrajectory()
    data_tracking_DLC = traitment_DLC.correction(data_DLC, data_tracking_DLC)

    norme_vecteur_entre_point = traitment_DLC.norme_vecteur_entre_points(data_tracking_DLC, name_pointsA='hang', name_pointsB='hand')
# ------------------------------------- save un csv du fichier corrigé de DLC ----------------------------------------
    name = f'\\testsave.csv'
    csv_save = SavingMethodes()
    csv_save.save_csv(path=dir_DLC + name, data=data_tracking_DLC)

# ------------------------------------- partie plot ---------------------------------------------------------
    from utilitaire_plot.TrajectoireBasicPlot import SpecifiquePlot
    plot_DLC = SpecifiquePlot()
    plot_DLC.plot_traj(data_tracking_DLC, data_DLC.items, 'DLC')

    print('fin')
