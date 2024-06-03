r"""
data.py

Utilities for processing of Data
"""
import random

import numpy as np
import torch
from nltk import word_tokenize
from scipy.io.wavfile import read
from torch.utils.data.dataset import Dataset

from src.model.layers import TacotronSTFT
from src.utilities.text import phonetise_text, text_to_sequence


def load_wav_to_torch(full_path):
    r"""
    Uses scipy to convert the wav file into torch tensor
    Args:
        full_path: "Wave location"

    Returns:
        torch.FloatTensor of wav data and sampling rate
    """
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split) for line in f if int(line.strip().split(split)[1]) == 0 or int(line.strip().split(split)[1]) == 2]
    return filepaths_and_text


class TextMelCollate:
    r"""
    Zero-pads model inputs and targets based on number of frames per setep
    """

    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        r"""
        Collate's training batch from normalized text and mel-spectrogram

        Args:
            batch (List): [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True
        )
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, : text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max(x[1].size(1) for x in batch)

        # include mel padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))

        langs = torch.zeros_like(input_lengths)
        speakers = torch.zeros((len(batch), batch[0][3].shape[0]))
        emos = torch.zeros((len(batch), batch[0][4].shape[0]))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, : mel.size(1)] = mel
            output_lengths[i] = mel.size(1)

            langs[i] = batch[ids_sorted_decreasing[i]][2]
            speakers[i, :] = batch[ids_sorted_decreasing[i]][3]
            emos[i, :] = batch[ids_sorted_decreasing[i]][4]
            
        # torch.empty is a substite for gate_padded, will be removed later when more
        # test ensures there is no regression
        return text_padded, input_lengths, mel_padded, torch.empty([1]), output_lengths, langs, speakers, emos


class TextMelLoader(Dataset):
    r"""
    Taken from Nvidia-Tacotron-2 implementation

    1) loads audio,text pairs
    2) normalizes text and converts them to sequences of one-hot vectors
    3) computes mel-spectrograms from audio files.
    """

    def __init__(self, audiopaths_and_text, hparams, transform=None):
        r"""
        Args:
            audiopaths_and_text:
            hparams:
            transform (list): list of transformation
        """
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.transform = transform
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.phonetise = hparams.phonetise
        self.cmu_phonetiser = hparams.cmu_phonetiser
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = TacotronSTFT(
            hparams.filter_length,
            hparams.hop_length,
            hparams.win_length,
            hparams.n_mel_channels,
            hparams.sampling_rate,
            hparams.mel_fmin,
            hparams.mel_fmax,
        )
        self.spk_embeds_path = hparams.spk_embeds_path
        self.emo_embeds_path = hparams.emo_embeds_path
        self.database_name_index = hparams.database_name_index
        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)

    def get_mel_text_pair(self, audiopath_and_text):
        r"""
        Takes audiopath_text list input where list[0] is location for wav file
            and list[1] is the text
        Args:
            audiopath_and_text (list): list of size 2
        """
        # separate filename and text (string)
        audiopath, lid, text = audiopath_and_text[0], audiopath_and_text[1], audiopath_and_text[2]
        filename = audiopath.split("/")[-1].split(".")[0]
        database_name = audiopath.split("/")[self.database_name_index]

        # This text is int tensor of the input representation
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        lang = self.get_lid(lid)

        speaker = torch.Tensor(np.load(f"{self.spk_embeds_path.replace('dataset_name', database_name)}/{filename}.npy"))
        emo = torch.Tensor(np.load(f"{self.spk_embeds_path.replace('dataset_name', database_name)}/{filename}.npy"))
        
        if self.transform:
            for t in self.transform:
                mel = t(mel)

        return (text, mel, lang, speaker, emo)

    def get_mel(self, filename):
        r"""
        Takes filename as input and returns its mel spectrogram
        Args:
            filename (string): Example: 'LJSpeech-1.1/wavs/LJ039-0212.wav'
        """
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError(f"{sampling_rate} SR doesn't match target {self.stft.sampling_rate} SR")
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, "Mel dimension mismatch: given {}, expected {}".format(
                melspec.size(0), self.stft.n_mel_channels
            )

        return melspec

    def get_text(self, text, lid):
        if self.phonetise:
            text = phonetise_text(self.cmu_phonetiser, text, word_tokenize)

        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners[lid]))
        return text_norm

    def get_lid(self, lid):
        lid = torch.IntTensor([int(lid)])
        return lid

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class Normalise:
    r"""
    Z-Score normalisation class / Standardisation class
    normalises the data with mean and std, when the data object is called

    Args:
        mean (int/tensor): Mean of the data
        std (int/tensor): Standard deviation
    """

    def __init__(self, mean, std):
        super().__init__()

        if not torch.is_tensor(mean):
            mean = torch.tensor(mean)
        if not torch.is_tensor(std):
            std = torch.tensor(std)

        self.mean = mean
        self.std = std

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        r"""
        Takes an input and normalises it

        Args:
            x (Any): Input to the normaliser

        Returns:
            (torch.FloatTensor): Normalised value
        """
        if not torch.is_tensor(x):
            x = torch.tensor(x)

        x = x.sub(self.mean).div(self.std)
        return x

    def inverse_normalise(self, x):
        r"""
        Takes an input and de-normalises it

        Args:
            x (Any): Input to the normaliser

        Returns:
            (torch.FloatTensor): Normalised value
        """
        if not torch.is_tensor(x):
            x = torch.tensor([x])

        x = x.mul(self.std).add(self.mean)
        return x

class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.
  
    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True, weights=None):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
        self.weights = weights
  
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
  
    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
  
        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i+1)
  
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket
  
    def __iter__(self):
      # deterministically shuffle based on epoch
      g = torch.Generator()
      g.manual_seed(self.epoch)
  
      indices = []
      if self.shuffle:
          for bucket in self.buckets:
              indices.append(torch.randperm(len(bucket), generator=g).tolist())
      else:
          for bucket in self.buckets:
              indices.append(list(range(len(bucket))))
  
      batches = []
      for i in range(len(self.buckets)):
          bucket = self.buckets[i]
          len_bucket = len(bucket)
          ids_bucket = indices[i]
          num_samples_bucket = self.num_samples_per_bucket[i]
  
          # add extra samples to make it evenly divisible
          rem = num_samples_bucket - len_bucket
          ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]
  
          # subsample
          ids_bucket = ids_bucket[self.rank::self.num_replicas]
  
          # batching
          for j in range(len(ids_bucket) // self.batch_size):
              batch = [bucket[idx] for idx in ids_bucket[j*self.batch_size:(j+1)*self.batch_size]]
              batches.append(batch)
  
      if self.shuffle:
          batch_ids = torch.randperm(len(batches), generator=g).tolist()
          batches = [batches[i] for i in batch_ids]
      self.batches = batches
  
      assert len(self.batches) * self.batch_size == self.num_samples
      return iter(self.batches)
  
    def _bisect(self, x, lo=0, hi=None):
      if hi is None:
          hi = len(self.boundaries) - 1
  
      if hi > lo:
          mid = (hi + lo) // 2
          if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
              return mid
          elif x <= self.boundaries[mid]:
              return self._bisect(x, lo, mid)
          else:
              return self._bisect(x, mid + 1, hi)
      else:
          return -1

    def __len__(self):
        return self.num_samples // self.batch_size