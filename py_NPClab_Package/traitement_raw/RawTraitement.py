from NPClab_Package.utilitaire_load import ImportNeuralynx, LoadData

import time as chrono
#
# class ImportRawSignal(RawSignals):
#     def __init__(self, csc_dir: str):
#         super().__init__(csc_dir)
#         self.utilitaire_neuralynx()


if __name__ == "__main__":
    t1 = chrono.time()

    dir_neuralynx: str = r'/data/cplx07 + bsl'
    Raw_brute = LoadData.init_data(ImportNeuralynx, dir_neuralynx, 'csc')

    t2 = chrono.time()
    print(f'temps global : {t2 - t1}')