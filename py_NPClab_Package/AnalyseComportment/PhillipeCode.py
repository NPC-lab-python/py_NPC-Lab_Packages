#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 1/12/2018 16:19

@author: phfaure2

%matplotlib qt pour plot sur fenetre separe

"""

import numpy as np
import random
import glob
from scipy import optimize
import pandas as pd
from py_NPClab_Package.utilitaire_traitement.Decorateur import mesure


###########  Classe

class fileexp():
    """ permet de definir la liste des fichiers """

    def __init__(self, genotype="WT", session="1", manip="Saline", repertoire="../Data-Complexity_All/d01",
                 listFichier=None):
        self.chemin = repertoire
        self.genotype = genotype
        self.manip = manip
        self.session = session
        if (listFichier != None):
            self.listfichier = listFichier
        else:
            self.loadDfile()

    def __repr__(self):
        """ representation quand on rentre objet dans interpreteur"""
        return "liste fichier({})".format(self.listfichier)

    def loadDfile(self):
        self.listfichier = glob.glob(self.chemin + "/" + "*traj.txt")


class experience():
    """ Classe définissant une experience caractérisé par
    - une souris
	- une session
    - un genotype de Souris par défault WT
    - un fichier de donnée (le fichier choix) on construit le nom du fichier trajectoire
    - une sequence de choix et une séquence de reward
    - la probabilité associé aux trois points (0,1,2). Par défaut 1,1,1

    ex s1=experience(fichier='/Users/phfaure2/ownCloud/SHARED_NCB-BU/data_fig(exp)/Data-Complexity_All/d10/wt1_cplx_10_17022017-1132.txt')
       s1.loadData()
    """

    def __init__(self, name="Souris", genotype="WT", session="1", manip="det", traitement="sal", fichier="Fichier.txt"):
        self.nameSouris = name
        self.genotype = genotype
        self.manip = manip
        self.session = session
        self.traitement = traitement
        self.fichierTraj = fichier
        self.fichier = self.fichierTraj.split("traj")[0] + self.fichierTraj.split("traj")[1]
        self.seqChoix = {}
        self.TimeChoix = {}
        self.seqReward = {}
        self.TimeReward = {}
        self.ProbaRew = [1, 1, 1]


    def __repr__(self):
        """ representation quand on rentre objet dans interpreteur"""
        return "experience:experience({}),session({}),nom({}),genotype({}),fichier({})".format(self.manip, self.session,
                                                                                               self.nameSouris,
                                                                                               self.genotype,
                                                                                               self.fichier)
    @mesure
    def load_Data(self):
        """
        temps actuel d'execution 0.58
        """
        dt_ms = 0.050
        time = 0.
        s = []
        s_time = []
        r = []
        r_time = []
        px = []
        py = []
        with open(self.fichierTraj, 'r') as lines:
            for i_line, line in enumerate(lines):
                if i_line == 0:
                    tt = line.replace(",", ".")
                    target = int(tt.split("\t")[2].split(".")[0])
                    px.append(float(tt.split("\t")[0]))
                    py.append(float(tt.split("\t")[1]))
                else:
                    tt = line.replace(",", ".")
                    px.append(float(tt.split("\t")[0]))
                    py.append(float(tt.split("\t")[1]))
                    prev_target = target
                    target = int(tt.split("\t")[2].split(".")[0])
                    time += dt_ms
                    if target != prev_target:
                        s.append(target)
                        s_time.append(time)

        self.seqChoix = np.transpose(s) #
        self.TimeChoix = np.transpose(s_time)

        self.trajX = np.transpose(px)
        self.trajY = np.transpose(py)

        with open(self.fichier, 'r') as lines:
            for i_line, line in enumerate(lines):
                target = int(line.split("\t")[1])
                targettime = int(line.split("\t")[0])
                r.append(target)
                r_time.append(targettime)

        self.seqReward = np.transpose(r)

        self.TimeReward = np.transpose(r_time)

        ##### filtrage des traj wd=5 rajouter qu'on enleve les mauvais points
        wd = 5
        nx = []
        ny = []
        nn = int((wd - 1) / 2)
        l = len(self.trajX)
        nx.extend(self.trajX[0:nn])
        ny.extend(self.trajY[0:nn])
        for i in range((nn + 1), (l - nn)):
            nx.append(np.mean(self.trajX[(i - nn - 1):(i + nn)]))
            ny.append(np.mean(self.trajY[(i - nn - 1):(i + nn)]))
        nx.extend(self.trajX[(l - nn - 1):l])
        ny.extend(self.trajY[(l - nn - 1):l])
        self.trajXF = np.transpose(nx)
        self.trajYF = np.transpose(ny)

    def distrib(self):
        """ proportion de choix A,B ou C """
        propList = [0, 0, 0]
        propList[0] = sum(self.seqChoix == 0) / len(self.seqChoix)
        propList[1] = sum(self.seqChoix == 1) / len(self.seqChoix)
        propList[2] = sum(self.seqChoix == 2) / len(self.seqChoix)
        return propList

    def retour(self):
        """ % de retour ex: seq ABA """
        ll = len(self.seqChoix)
        s1 = self.seqChoix[0:ll - 2]
        s2 = self.seqChoix[2:ll]
        nbreturn = sum(np.array(s1) - np.array(s2) == 0)
        return nbreturn / (ll - 2)

    def successrate(self):
        """ % de retour ex: seq ABA
        self.seqChoix correspond au  point par frame donc 6001 poour 5min
        self.seqReward correspond au reward contenue dans le fichier txt reward
        """
        ll = len(self.seqChoix)
        ll2 = len(self.seqReward)
        return ll2 / ll

    def timetogoal(self):
        """ mean time between two targets  """
        tt = np.diff(self.TimeChoix)
        return np.mean(tt)

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


class groupexperience():
    """
    Permet de definir parametres manip
    Manages a group of animals.
    le fichier est passé via fileexp
    filewt=fileexp(repertoire=pathr,session=i)
    filewt.loadDfile()
    pooli=groupexperience()
    pooli.fichiermanip=filewt.listfichier
    """

    def __init__(self, genotype="WT", manip="complex", session="1", traitement="sal"):
        self.genotype = genotype
        self.manip = manip
        self.session = session
        self.traitement = traitement
        self.fichiermanip = fileexp()  # fichier contenant la liste des fichiers du groupe
        self.animalList = {}

    def loadAnimal(self, name="Souris"):
        for i_line, fic in enumerate(self.fichiermanip):
            nomsouris = fic.split("/")[-1].split("_")[0]
            self.animalList[i_line] = experience(fichier=fic, session=self.session, genotype=self.genotype,
                                                 manip=self.manip, name=nomsouris, traitement=self.traitement)

    def loadData(self):
        for animal in self.animalList.keys():
            self.animalList[animal].loadData()

    def tabData(self):
        p1 = []
        p2 = []
        p3 = []
        ret = []
        name = []
        fich = []
        p100f25 = []
        p100f50 = []
        p50f25 = []
        beta = []
        betap = []
        phi = []
        srate = []
        tgoal = []
        for j in self.animalList.keys():
            ret.append(self.animalList[j].retour())
            fich.append(self.animalList[j].fichier.split("/")[-1])
            name.append(self.animalList[j].nameSouris)
            srate.append(self.animalList[j].successrate())
            tgoal.append(self.animalList[j].timetogoal())
            aa, bb, cc = self.animalList[j].distrib()
            if self.animalList[j].ProbaRew != [1, 1, 1]:
                tt = [aa, bb, cc]
                aa = tt[self.animalList[j].ProbaRew.index(0.25)]
                bb = tt[self.animalList[j].ProbaRew.index(0.5)]
                cc = tt[self.animalList[j].ProbaRew.index(1)]
            p1.append(aa)
            p2.append(bb)
            p3.append(cc)
            p100f25.append(self.animalList[j].matrixchoix()[0, 2])
            p100f50.append(self.animalList[j].matrixchoix()[1, 2])
            p50f25.append(self.animalList[j].matrixchoix()[2, 1])
            beta.append(self.animalList[j].beta().x[0])
            betap.append(self.animalList[j].betaphi().x[0])
            phi.append(self.animalList[j].betaphi().x[1])

        DataFr = pd.DataFrame({'Nom': name,
                               'Fichier': fich,
                               'probaA': p1,
                               'probaB': p2,
                               'probaC': p3,
                               'p100f25': p100f25,
                               'p100f50': p100f50,
                               'p50f25': p50f25,
                               'ret': ret,
                               'tGoal': tgoal,
                               'sucess': srate,
                               'beta': beta,
                               'betaphy': betap,
                               'phy': phi})
        return (DataFr)




if __name__ == '__main__':
    # import numpy as np
    # import pandas as pd
    # from py_NPClab_Package.utilitaire_load.basic_load import LabviewFilesReward, LabviewFilesTrajectory, LoadData
    # from py_NPClab_Package.utilitaire_traitement.TrajectoryTraitement import BasicTraitmentTrajectory
    # from py_NPClab_Package.traitement_labview.Labview_traitment import AnalyseFromLabview
    #
    #
    # dir_profile_pattern: str = r'Y:\Analyse_maxime\profile_pattern'
    #
    # # dir_save: str = r'Y:\Analyse_maxime\cplx25\save'
    # # dir_data: str = r'Y:\Analyse_maxime\cplx25'
    # # dir_spikefile: str = r'Y:\Analyse_maxime\cplx25\clustering\*.txt'
    # # dir_txt_traj: str = r'Y:\Analyse_maxime\cplx25\fichier_traj\*.txt'
    #
    # num_segment = 1
    # name_neurone: str = 'segment1_SpikeFile_CSC2_CSC6_CSC7_CSC10_SS_01'
    #
    # dir_global = r'Y:\Analyse_maxime'
    #
    # # ----- det
    # dir_save: str = r'Y:\Analyse_maxime\cplx07 + bsl\save'
    # dir_data: str = r'Y:\Analyse_maxime\cplx07 + bsl'
    # dir_spikefile: str = r'Y:\Analyse_maxime\cplx07 + bsl\clustering\*.txt'
    # dir_txt_traj: str = r'Y:\Analyse_maxime\cplx07 + bsl\fichier_traj\*.txt'
    #
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
    #
    # # ------------------------------------- parti traitement de la trajectoire labview -----------------
    # traitment_AFL = BasicTraitmentTrajectory()
    # data_traiter_AFL = traitment_AFL.correction(data_AFL, data_AFL.format_correction)
    exp = experience(name="agam3_equdiff_9_06072017-1342traj", genotype="WT", session="1", manip="proba", traitement="nic",
                     fichier=r'Y:\codephilippe\data\stress\NicDay9-10proba\agam3_equdiff_9_06072017-1342traj.txt')
    exp.load_Data()
    exp.matrixchoix()
    print('r')