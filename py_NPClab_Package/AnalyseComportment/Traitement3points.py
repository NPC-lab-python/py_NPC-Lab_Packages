import numpy as np
import random
import glob
from scipy import optimize
import pandas as pd


class AnalyseProbabiliste(object):

    def matrixchoix(self):
        """ % matrice de transition normalisé"""
        mata = np.zeros((3, 3))

        if self.ProbaRew == [1, 1, 1]:
            sequ = self.seqChoix
        else:
            sequ = self.seqChoix + 3
            sequ[sequ == (self.ProbaRew.index(0.25) + 3)] = 0
            sequ[sequ == (self.ProbaRew.index(0.5) + 3)] = 1
            sequ[sequ == (self.ProbaRew.index(1) + 3)] = 2
        ll = len(sequ)
        s1 = sequ[0:ll - 1]
        s2 = sequ[1:ll]
        for i in range(0, (ll - 1)):
            mata[s1[i], s2[i]] += 1
        mata[0, :] = mata[0, :] / np.sum(mata[0, :])
        mata[1, :] = mata[1, :] / np.sum(mata[1, :])
        mata[2, :] = mata[2, :] / np.sum(mata[2, :])
        return mata


    def betaphi(self):
        """Estimation de beta et phi"""
        """calcul de la matrice de choix """
        mata = np.zeros((3, 3))
        ll = len(self.seqChoix)
        s1 = self.seqChoix[0:ll - 1]
        s2 = self.seqChoix[1:ll]
        for i in range(0, (ll - 1)):
            mata[s1[i], s2[i]] += 1
        mata[0, :] = mata[0, :] / np.sum(mata[0, :])
        mata[1, :] = mata[1, :] / np.sum(mata[1, :])
        mata[2, :] = mata[2, :] / np.sum(mata[2, :])

        expe = mata

        """ le set de probas """
        vec = self.ProbaRew

        """ exemple de conditions initiales (1,1) est ok pour les fits beta,phi"""
        x0 = np.array([1, 0])

        def f(params):
            a = np.array([np.exp(params[0] * (vec[0] + params[1] * vec[0] * (1 - vec[0])))])
            b = np.array([np.exp(params[0] * (vec[1] + params[1] * vec[1] * (1 - vec[1])))])
            c = np.array([np.exp(params[0] * (vec[2] + params[1] * vec[2] * (1 - vec[2])))])
            model = ([[0, b / (b + c), c / (b + c)], [a / (a + c), 0, c / (a + c)], [a / (a + b), b / (a + b), 0]])
            LL = -np.log(np.prod(model ** expe));
            return LL

        """ fonction minimize """
        bnds = ((2e-16, 10), (2e-16, 4))
        minimum = optimize.minimize(f, x0, method='L-BFGS-B', bounds=bnds)

        return (minimum)


    def beta(self):
        """Estimation de beta"""
        """calcul de la matrice de choix """
        mata = np.zeros((3, 3))
        ll = len(self.seqChoix)
        s1 = self.seqChoix[0:ll - 1]
        s2 = self.seqChoix[1:ll]
        for i in range(0, (ll - 1)):
            mata[s1[i], s2[i]] += 1
        mata[0, :] = mata[0, :] / np.sum(mata[0, :])
        mata[1, :] = mata[1, :] / np.sum(mata[1, :])
        mata[2, :] = mata[2, :] / np.sum(mata[2, :])

        expe = mata

        """ le set de probas """
        vec = self.ProbaRew

        """ exemple de conditions initiales (1,1) est ok pour les fits beta,phi"""
        x0 = np.array([1])

        def f(params):
            a = np.array([np.exp(params[0] * vec[0])])
            b = np.array([np.exp(params[0] * vec[1])])
            c = np.array([np.exp(params[0] * vec[2])])
            model = ([[0, b / (b + c), c / (b + c)], [a / (a + c), 0, c / (a + c)], [a / (a + b), b / (a + b), 0]])
            LL = -np.log(np.prod(model ** expe));
            return LL

        """ fonction minimize"""

        bnds = ((2e-16, 10),)
        minimum = optimize.minimize(f, x0, method='L-BFGS-B', bounds=bnds)

        return (minimum)
