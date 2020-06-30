import time
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def mesure(fonction):

    def mesure_du_temps(*args, **kwargs):
        start_time = time.perf_counter()
        result = fonction(*args, **kwargs)
        stop_time = time.perf_counter()
        total = stop_time - start_time
        logging.debug(f"temps d'execution : {total} secondes {fonction.__name__}")
        return result
    return mesure_du_temps