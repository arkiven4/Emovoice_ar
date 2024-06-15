# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import torch.nn.functional as F
import torch.nn as nn
from typing import Optional

import torch
from librosa.filters import mel as librosa_mel_fn
from audio_processing import dynamic_range_compression, dynamic_range_decompression
from stft import STFT


class TacotronSTFT(torch.nn.Module):
    """Short Time Fourier Transformation."""

    def __init__(
        self,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        sampling_rate=22050,
        mel_fmin=0.0,
        mel_fmax=8000.0,
    ):
        super().__init__()
        self.n_mel_channels = n_mel_channels  # 80
        self.sampling_rate = sampling_rate  # 22050
        self.stft_fn = STFT(filter_length, hop_length, win_length)  # default values
        # """This produces a linear transformation matrix to project FFT bins onto Mel-frequency bins."""
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax
        )  # all default values

        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert torch.min(y.data) >= -1
        assert torch.max(y.data) <= 1

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output

"""functions.py.

File for custom utility functions to improve numerical precision
"""
import torch


def log_clamped(x, eps=1e-04):
    clamped_x = torch.clamp(x, min=eps)
    return torch.log(clamped_x)


def inverse_sigmod(x):
    r"""
    Inverse of the sigmoid function
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    return log_clamped(x / (1.0 - x))


def inverse_softplus(x):
    r"""
    Inverse of the softplus function
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    return log_clamped(torch.exp(x) - 1.0)


def logsumexp(x, dim):
    r"""
    Differentiable LogSumExp: Does not creates nan gradients
        when all the inputs are -inf
    Args:
        x : torch.Tensor -  The input tensor
        dim: int - The dimension on which the log sum exp has to be applied
    """

    m, _ = x.max(dim=dim)
    mask = m == -float("inf")

    s = (x - m.masked_fill_(mask, 0).unsqueeze(dim=dim)).exp().sum(dim=dim)
    return s.masked_fill_(mask, 1).log() + m.masked_fill_(mask, -float("inf"))


def log_domain_matmul(log_a, log_b):
    r"""
    Multiply two matrices in log domain
    Args:
        log_a : m x n
        lob_b : n x p
        out : m x p
    Returns:
        Computes output_{i, j} = logsumexp_k [ log_A_{i, k} + log_B{k, j} ]
    """

    m, n, p = log_a.shape[0], log_a.shape[1], log_b.shape[1]

    # Dimensions must be same to add

    # Expand A to the p size
    log_A_expanded = log_a.unsqueeze(2).expand((m, n, p))
    # Expand B to m size
    log_B_expanded = log_b.unsqueeze(0).expand((m, n, p))
    # These expansion will result in addition

    elementwise_sum = log_A_expanded + log_B_expanded

    out = logsumexp(elementwise_sum, 1)

    return out


def masked_softmax(vec, dim=0):
    r"""Outputs masked softmax"""
    mask = ~torch.eq(vec, 0)
    exps = torch.exp(vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True)
    softmax_values = masked_exps / masked_sums
    return softmax_values


def masked_log_softmax(vec, dim=0):
    r"""Outputs masked log_softmax"""
    mask = ~torch.eq(vec, 0)
    exps = torch.exp(vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True)
    softmax_values = masked_exps / masked_sums
    idx = softmax_values != 0
    softmax_values[idx] = torch.log(softmax_values[idx])
    return softmax_values


def get_mask_from_len(lengths, max_len=None, device="cpu", out_tensor=None, dtype=None):
    if max_len is None:
        max_len = lengths.max().item()
    ids = (
        torch.arange(max_len, device=device, dtype=dtype)
        if out_tensor is None
        else torch.arange(max_len, out=out_tensor)
    )
    mask = ids < lengths.unsqueeze(1)
    return mask


def get_mask_for_last_item(lengths, device="cpu", out_tensor=None):
    """Returns n-1 mask for the last item in the sequence.

    Args:
        lengths (torch.IntTensor): lengths in a batch
        device (str, optional): Defaults to "cpu".
        out_tensor (torch.Tensor, optional): uses the memory of a specific tensor.
            Defaults to None.
    """
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=device) if out_tensor is None else torch.arange(0, max_len, out=out_tensor)
    mask = ids == lengths.unsqueeze(1) - 1
    return mask


######################################################
# Begin Glow-TTS methods
# https://github.com/jaywalnut310/glow-tts
######################################################


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def squeeze(x, x_mask=None, n_sqz=2):
    b, c, t = x.size()

    t = (t // n_sqz) * n_sqz
    x = x[:, :, :t]
    x_sqz = x.view(b, c, t // n_sqz, n_sqz)
    x_sqz = x_sqz.permute(0, 3, 1, 2).contiguous().view(b, c * n_sqz, t // n_sqz)

    if x_mask is not None:
        x_mask = x_mask[:, :, n_sqz - 1 :: n_sqz]
    else:
        x_mask = torch.ones(b, 1, t // n_sqz).to(device=x.device, dtype=x.dtype)
    return x_sqz * x_mask, x_mask


def unsqueeze(x, x_mask=None, n_sqz=2):
    b, c, t = x.size()

    x_unsqz = x.view(b, n_sqz, c // n_sqz, t)
    x_unsqz = x_unsqz.permute(0, 2, 3, 1).contiguous().view(b, c // n_sqz, t * n_sqz)

    if x_mask is not None:
        x_mask = x_mask.unsqueeze(-1).repeat(1, 1, 1, n_sqz).view(b, 1, t * n_sqz)
    else:
        x_mask = torch.ones(b, 1, t * n_sqz).to(device=x.device, dtype=x.dtype)
    return x_unsqz * x_mask, x_mask

def to_gpu(x):
    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)