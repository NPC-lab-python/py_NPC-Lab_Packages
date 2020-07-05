# -*-coding:utf-8 -*-

from py_NPClab_Package.utilitaire_plot.BasicPlotSpike import GenericPlotV2

plot = GenericPlotV2()

from py_NPClab_Package.traitement_global.TraitementNeuroneGlobal import ConstructRaster

dir_data: str = r'Y:\Analyse_maxime\global_raster'

global_data = ConstructRaster()

reward_all, omission_all, vecpoints = global_data.global_load(path=r'Y:\Analyse_maxime\global_raster')
#
plot.plot_kernel_denty_global(name=f'{dir_data}\\reward_all', around_event=reward_all, x=vecpoints.iloc[0])

plot.plot_kernel_denty_global(name=f'{dir_data}\\omission_all', around_event=omission_all, x=vecpoints.iloc[0])

dir_data: str = r'Y:\Analyse_maxime\global_raster_det'

global_data = ConstructRaster()

reward_all, omission_all, vecpoints = global_data.global_load(path=r'Y:\Analyse_maxime\global_raster_det')
#
plot.plot_kernel_denty_global(name=f'{dir_data}\\reward_all', around_event=reward_all, x=vecpoints.iloc[0])