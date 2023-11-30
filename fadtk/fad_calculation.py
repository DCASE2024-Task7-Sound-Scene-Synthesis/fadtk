from argparse import ArgumentParser

from .fad import FrechetAudioDistance, log
from .model_loader import *
from .fad_batch import cache_embedding_files

def calculate_fad(model_type, baseline, eval, workers=8):

    models = {m.name: m for m in get_all_models()}
    print(models.keys())
    model = models[model_type]
    for d in [baseline, eval]:
        if Path(d).is_dir():
            cache_embedding_files(d, model, workers=workers)

    fad = FrechetAudioDistance(model, audio_load_worker=workers, load_model=False)

    score = fad.score(baseline, eval)

    return(score)