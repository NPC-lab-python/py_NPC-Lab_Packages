import unittest
from traitement_labview.Labview_traitment import AnimalSurface, ImportData, AnalyseFromLabview
from pandas import DataFrame, Series
import pandas as pd
import numpy as np
from typing import Tuple
from numpy.core.multiarray import ndarray

Trajectoire = Tuple[ndarray]

from numpy.core.multiarray import ndarray


class MyTestImportData(unittest.TestCase):


    def test_calcul_norme_(self):
        """
        A(2;3) B(5;1) C(4;4)
        :return:
        """

        x = pd.Series([2, 5, 4], name='x')
        y = pd.Series([3, 1, 4], name='y')

        x1 = x[0]
        x2 = x[1]
        y1 = y[0]
        y2 = y[1]

        norme_vecteur = ImportData('')._calcul_norme_(x1, x2, y1, y2)
        self.assertEqual(np.around(norme_vecteur[0], 1), 3.60)
        self.assertIsInstance(norme_vecteur, Series)

    def test_coordonnee_vecteur_(self):
        """
        A(2;3) B(5;1) C(4;4)
        :return:
        """

        x = pd.Series([2,5,4], name='x')
        y = pd.Series([3,1,4], name='y')

        x1 = x[0]
        x2 = x[1]
        y1 = y[0]
        y2 = y[1]

        test = pd.Series([3,-2],name='coordonnee_vecteur',dtype=int)

        coordonnee_vecteur = ImportData('')._coordonnee_vecteur_(x1, x2, y1, y2)
        self.assertEqual(coordonnee_vecteur.loc[0], test.loc[0])
        self.assertEqual(coordonnee_vecteur.loc[1], test.loc[1])

        self.assertIsInstance(coordonnee_vecteur, Series)

    def test_calcul_norme_vecteur_(self):
        """

        :return:
        """
        x = pd.Series([2, 5, 4], name='x')
        y = pd.Series([3, 1, 4], name='y')

        data_test = pd.DataFrame({'Ax': 2, 'Ay': 3, 'Bx': 5, 'By': 1, 'Cx': 4, 'Cy': 4}, index=[0])

        items = list(data_test)
        couple_pointA = [items[0], items[1]]
        couple_pointB = [items[2], items[3]]

        test = pd.Series([3, -2], name='coordonnee_vecteur', dtype=int)

        norme_coordo = ImportData('')._norme_vecteur_(data=data_test, couple_pointsA=couple_pointA, couple_pointsB=couple_pointB)

        self.assertIsInstance(norme_coordo, tuple)
        self.assertEqual(norme_coordo[1].loc[0], test.loc[0])
        self.assertEqual(norme_coordo[1].loc[1], test.loc[1])

        self.assertIsInstance(norme_coordo[1], Series)
        self.assertIsInstance(norme_coordo[0], Series)

    def test_norme_vecteur_entre_points(self):

        data_test = pd.DataFrame({'Ax': 2, 'Ay': 3, 'Bx': 5, 'By': 1, 'Cx': 4, 'Cy': 4}, index=[0])

        items = list(data_test)
        couple_pointA = [items[0], items[1]]
        couple_pointB = [items[2], items[3]]

        test = pd.Series([3, -2], name='coordonnee_vecteur', dtype=int)

        norme_coordo = ImportData('').norme_vecteur_entre_points(data=data_test, couple_pointsA=couple_pointA, couple_pointsB=couple_pointB)

        self.assertIsInstance(norme_coordo, tuple)
        self.assertEqual(norme_coordo[1].loc[0], test.loc[0])
        self.assertEqual(norme_coordo[1].loc[1], test.loc[1])

        self.assertIsInstance(norme_coordo[1], Series)
        self.assertIsInstance(norme_coordo[0], Series)

#
class MyTestAninmalSurface(unittest.TestCase):

    def test_calcul_ps_ABAC(self):
        # AB: Series = pd.Series(3.6)
        # AC: Series = pd.Series(2.24)
        # BC: Series = pd.Series(3.16)
        AB = pd.Series([3, -2], name='coordonnee_vecteur', dtype=int)
        AC = pd.Series([2, 1], name='coordonnee_vecteur', dtype=int)
        x = pd.Series(AB[0])
        x_p = pd.Series(AC[0])
        y = pd.Series(AB[1])
        y_p = pd.Series(AC[1])

        abac = AnimalSurface()._calcul_ps_ABAC_(x, x_p, y, y_p)
        if len(abac) == len(x):
            i = True
        self.assertTrue(i)
        self.assertIsInstance(abac, Series)
        self.assertEqual(abac.loc[0], 4)

class TestAnalyseFromLabview(unittest.TestCase):

    def test_make_data_from_traj_(self):
        trajectoire: tuple = (25.3, 56.5, 0)
        event_time1: Series = pd.Series(trajectoire[0], name='trajectoire_x')
        event_time2: Series = pd.Series(trajectoire[1], name='trajectoire_y')
        event_time3: Series = pd.Series(trajectoire[2], name='reward')

        data_test: DataFrame = pd.DataFrame(
            {event_time1.name: event_time1, event_time2.name: event_time2, event_time3.name: event_time3})

        name_test = pd.Index(['trajectoire_x', 'trajectoire_y', 'reward'])
        data = AnalyseFromLabview()._make_data_from_traj_(trajectoire)
        self.assertIsInstance(data, DataFrame)
        self.assertEqual(data.columns.array[:], name_test.array[:])
        self.assertListEqual(list(data['trajectoire_x']), list(event_time1))
        self.assertListEqual(list(data['trajectoire_y']), list(event_time2))
        self.assertListEqual(list(data['reward']), list(event_time3))

    def test_search_reward_from_traj_(self):
        data_set: Series = pd.Series([0, 0, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 2, 2, 2], name='reward', dtype=int)
        reward: int = 4
        data: Series = AnalyseFromLabview()._search_reward_from_traj_(data_set)
        self.assertIsInstance(data, ndarray)
        self.assertEqual(len(data), reward)

    def test_make_basetime_from_traj_(self):
        base_time: Series = pd.Series(np.arange(0, 50 * 50, 50))
        time_event_x: Series = pd.Series(pd.RangeIndex(0, 50, 1), dtype=int)
        time_event_y: Series = pd.Series(0, pd.RangeIndex(0, 50, 1), dtype=int)

        data_set: Trajectoire = tuple([np.array(np.random.random_sample(50)),
                                       np.array(np.random.random_sample(50)),
                                       np.array(np.random.random_sample(50))])

        time_structure_set: DataFrame = pd.DataFrame({'base_time': base_time,
                                                      'time_event_x': time_event_x,
                                                      'time_event_y': time_event_y}, dtype=int)

        name_test = pd.Index(['base_time', 'time_event_x', 'time_event_y'])
        data = AnalyseFromLabview()._make_basetime_from_traj_(data_set)
        self.assertEqual(data.columns.array[:], name_test.array[:])
        self.assertListEqual(list(data['base_time']), list(time_structure_set['base_time']))
        self.assertListEqual(list(data['time_event_x']), list(time_structure_set['time_event_x']))
        self.assertListEqual(list(data['time_event_y']), list(time_structure_set['time_event_y']))

if __name__ == '__main__':
    unittest.main()
