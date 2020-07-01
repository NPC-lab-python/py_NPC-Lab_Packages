from py_NPClab_Package.utlilitaire_saving.Saving_traitment import SaveSerialisation
from pandas import DataFrame, Series

class SaveRasterData(SaveSerialisation):
    def __init__(self):
        super().__init__()

    def save_raster(self, name_data: str, dir_save: str, data: DataFrame):
        self._set_conf_(name=name_data, dir_save_conf=dir_save, data=data)