"""
Calculate Frechet Audio Distance betweeen two audio directories.

Frechet distance implementation adapted from: https://github.com/mseitzer/pytorch-fid

VGGish adapted from: https://github.com/harritaylor/torchvggish
"""
import os
import numpy as np
import torch
from torch import nn
from scipy import linalg
from tqdm import tqdm
import soundfile as sf
import resampy
from multiprocessing.dummy import Pool as ThreadPool
from pathlib import Path
from hypy_utils.tqdm_utils import tq, tmap, pmap, smap

from .models.pann import Cnn14_16k


SAMPLE_RATE = 16000


def load_audio_task(fname):
    wav_data, sr = sf.read(fname, dtype='int16')
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    wav_data = wav_data / 32768.0  # Convert to [-1.0, +1.0]

    # Convert to mono
    if len(wav_data.shape) > 1:
        wav_data = np.mean(wav_data, axis=1)

    if sr != SAMPLE_RATE:
        wav_data = resampy.resample(wav_data, sr, SAMPLE_RATE)

    return wav_data


class FrechetAudioDistance:
    def __init__(self, model_name="vggish", use_pca=False, use_activation=False, verbose=False, audio_load_worker=8):
        self.model_name = model_name
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.__get_model(model_name=model_name, use_pca=use_pca, use_activation=use_activation)
        self.verbose = verbose
        self.audio_load_worker = audio_load_worker
    
    def __get_model(self, model_name="vggish", use_pca=False, use_activation=False):
        """
        Params:
        -- x   : Either 
            (i) a string which is the directory of a set of audio files, or
            (ii) a np.ndarray of shape (num_samples, sample_length)
        """
        if model_name == "vggish":
            # S. Hershey et al., "CNN Architectures for Large-Scale Audio Classification", ICASSP 2017
            self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
            if not use_pca:
                self.model.postprocess = False
            if not use_activation:
                self.model.embeddings = nn.Sequential(*list(self.model.embeddings.children())[:-1])
        
        elif model_name == "pann":
            # Kong et al., "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition", IEEE/ACM Transactions on Audio, Speech, and Language Processing 28 (2020)
            model_path = os.path.join(torch.hub.get_dir(), "Cnn14_16k_mAP%3D0.438.pth")
            if not(os.path.exists(model_path)):
                torch.hub.download_url_to_file('https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth', torch.hub.get_dir())
            self.model = Cnn14_16k(sample_rate=16000, window_size=512, hop_size=160, mel_bins=64, fmin=50, fmax=8000, classes_num=527)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])

        self.model.eval()

    def cache_embedding_file(self, audio_dir: str | Path, sr=SAMPLE_RATE) -> np.ndarray:
        """
        Compute embedding for an audio file and cache it to a file.
        """
        audio_dir = Path(audio_dir)
        cache = audio_dir.parent / "embeddings" / audio_dir.with_suffix(".npy").name

        if cache.exists():
            return np.load(cache)

        # Load file
        wav_data = load_audio_task(audio_dir)
        
        # Compute embedding
        embd = self.get_embedding(wav_data, sr)
        
        # Save embedding
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, embd)

        return embd

    def get_embedding(self, audio: np.ndarray, sr=SAMPLE_RATE) -> np.ndarray:
        """
        Get embedding for one audio sample
        """
        if self.model_name == "vggish":
            embd = self.model.forward(audio, sr)
        elif self.model_name == "pann":
            with torch.no_grad():
                out = self.model(torch.tensor(audio).float().unsqueeze(0), None)
                embd = out['embedding'].data[0]
        if self.device == torch.device('cuda'):
            embd = embd.cpu()
        embd = embd.detach().numpy()

        return embd
    
    def calculate_embd_statistics(self, embd_lst):
        if isinstance(embd_lst, list):
            embd_lst = np.array(embd_lst)
        mu = np.mean(embd_lst, axis=0)
        sigma = np.cov(embd_lst, rowvar=False)
        return mu, sigma
    
    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        Adapted from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
        
        Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)
    
    def get_embeddings_files(self, dir: str | Path):
        """
        Get embeddings for all audio files in a directory.
        """
        dir = Path(dir)

        # List valid audio files
        files = [dir / f for f in os.listdir(dir) if f.endswith(".wav")]
        files = [f for f in files if f.is_file()]

        if self.verbose:
            print(f"[Frechet Audio Distance] Loading {len(files)} audio files from {dir}...")

        # Map load task
        # embd_lst = tmap(self.cache_embedding_file, files, disable=(not self.verbose), desc="Loading audio files...")
        embd_lst = smap(self.cache_embedding_file, files)

        return np.concatenate(embd_lst, axis=0)

    def score(self, background_dir, eval_dir):
        try:
            embds_background = self.get_embeddings_files(background_dir)
            embds_eval = self.get_embeddings_files(eval_dir)

            if len(embds_background) == 0 or len(embds_eval) == 0:
                print("[Frechet Audio Distance] background or eval set is empty, exitting...")
                return -1
            
            mu_background, sigma_background = self.calculate_embd_statistics(embds_background)
            mu_eval, sigma_eval = self.calculate_embd_statistics(embds_eval)

            fad_score = self.calculate_frechet_distance(
                mu_background, 
                sigma_background, 
                mu_eval, 
                sigma_eval
            )

            return fad_score
            
        except Exception as e:
            print("[Frechet Audio Distance] exception thrown, {}".format(str(e)))
            raise e