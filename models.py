from math import sqrt

import torch
from torch import nn
from torch.nn import functional as F

from HMMModel.HMM import HMM
from modules import LinearReluInitNorm, ConvNorm, ActNorm, InvConvNear, CouplingBlock

from commons import get_mask_from_len, squeeze, unsqueeze

class Prenet(nn.Module):
    r"""
    MLP prenet module
    """

    def __init__(self, in_dim, n_layers, prenet_dim, prenet_dropout):
        super().__init__()
        in_sizes = [in_dim] + [prenet_dim for _ in range(n_layers)]
        self.prenet_dropout = prenet_dropout
        self.layers = nn.ModuleList(
            [
                LinearReluInitNorm(in_size, out_size, bias=False)
                for (in_size, out_size) in zip(in_sizes[:-1], in_sizes[1:])
            ]
        )

    def forward(self, x, dropout_flag):
        for linear in self.layers:
            x = F.dropout(linear(x), p=self.prenet_dropout, training=dropout_flag)
        return x

class Encoder(nn.Module):
    """Encoder module:

    - Three 1-d convolution banks
    - Bidirectional LSTM
    """

    def __init__(self, hparams):
        super().__init__()

        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.state_per_phone = hparams.state_per_phone

        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(
                    hparams.encoder_embedding_dim,
                    hparams.encoder_embedding_dim,
                    kernel_size=hparams.encoder_kernel_size,
                    stride=1,
                    padding=int((hparams.encoder_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="relu",
                ),
                nn.BatchNorm1d(hparams.encoder_embedding_dim),
            )
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(
            hparams.encoder_embedding_dim,
            int(hparams.encoder_embedding_dim / 2) * hparams.state_per_phone,
            1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x, input_lengths):
        r"""
        Takes embeddings as inputs and returns encoder representation of them
        Args:
            x (torch.float) : input shape (32, 512, 139)
            input_lengths (torch.int) : (32)

        Returns:
            outputs (torch.float):
                shape (batch, text_len * phone_per_state, encoder_embedding_dim)
            input_lengths (torch.float): shape (batch)
        """

        batch_size = x.shape[0]
        t_len = x.shape[2]

        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths_np = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths_np, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)  # We do not use the hidden or cell states

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        outputs = outputs.reshape(batch_size, t_len * self.state_per_phone, self.encoder_embedding_dim)
        input_lengths = input_lengths * self.state_per_phone

        return outputs, input_lengths  # (32, 139, 519)

class FlowSpecDecoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.in_channels = hparams.n_mel_channels
        self.hidden_channels = hparams.flow_hidden_channels
        self.kernel_size = hparams.kernel_size_dec
        self.dilation_rate = hparams.dilation_rate
        self.n_blocks = hparams.n_blocks_dec
        self.n_layers = hparams.n_block_layers
        self.p_dropout = hparams.p_dropout_dec
        self.n_split = hparams.n_split
        self.n_sqz = hparams.n_sqz
        self.sigmoid_scale = hparams.sigmoid_scale
        self.gin_channels = hparams.gin_channels
        self.emoin_channels = hparams.emoin_channels

        self.flows = nn.ModuleList() 
        for b in range(hparams.n_blocks_dec):
            self.flows.append(ActNorm(channels=hparams.n_mel_channels * hparams.n_sqz))
            self.flows.append(InvConvNear(channels=hparams.n_mel_channels * hparams.n_sqz, n_split=hparams.n_split))
            self.flows.append(
                CouplingBlock(
                    hparams.n_mel_channels * hparams.n_sqz,
                    hparams.flow_hidden_channels,
                    kernel_size=hparams.kernel_size_dec,
                    dilation_rate=hparams.dilation_rate,
                    n_layers=hparams.n_block_layers,
                    gin_channels=hparams.gin_channels,
                    emoin_channels=hparams.emoin_channels,
                    p_dropout=hparams.p_dropout_dec,
                    sigmoid_scale=hparams.sigmoid_scale,
                )
            )

    def forward(self, x, x_lengths, g=None, emo=None, reverse=False):
        """Calls Glow-TTS decoder

        Args:
            x (torch.FloatTensor): Input tensor (batch_len, n_mel_channels, T_max)
            x_lengths (torch.IntTensor): lens of mel spectrograms (batch_len)
            g (_type_, optional): _description_. Defaults to None.
            reverse (bool, optional): True when synthesising. Defaults to False.

        Returns:
            _type_: _description_
        """
        assert x.shape == (
            x_lengths.shape[0],
            self.in_channels,
            x_lengths.max(),
        ), f"The shape of the  \
            input should be (batch_dim, n_mel_channels, T_max) but received {x.shape}"
        x, x_lengths, x_max_length = self.preprocess(x, x_lengths, x_lengths.max())

        x_mask = get_mask_from_len(x_lengths, x_max_length, device=x.device, dtype=x.dtype).unsqueeze(1)

        if not reverse:
            flows = self.flows
            logdet_tot = 0
        else:
            flows = reversed(self.flows)
            logdet_tot = None

        if self.n_sqz > 1:
            x, x_mask = squeeze(x, x_mask, self.n_sqz)

        for f in flows:
            if not reverse:
                x, logdet = f(x, x_mask, g=g, emo=emo, reverse=reverse)
                logdet_tot += logdet
            else:
                x, logdet = f(x, x_mask, g=g, emo=emo, reverse=reverse)
        if self.n_sqz > 1:
            x, x_mask = unsqueeze(x, x_mask, self.n_sqz)
        return x, x_lengths, logdet_tot

    def store_inverse(self):
        for f in self.flows:
            f.store_inverse()

    def preprocess(self, y, y_lengths, y_max_length):
        if y_max_length is not None:
            y_max_length = torch.div(y_max_length, self.n_sqz, rounding_mode="floor") * self.n_sqz
            y = y[:, :, :y_max_length]
        y_lengths = torch.div(y_lengths, self.n_sqz, rounding_mode="floor") * self.n_sqz
        return y, y_lengths, y_max_length

class OverFlow(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.embedding = nn.Embedding(hparams.n_symbols, hparams.symbols_embedding_dim)
        if hparams.warm_start or (hparams.checkpoint_path is None):
            # If warm start or resuming training do not re-initialize embeddings
            std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
            val = sqrt(3.0) * std  # uniform bounds for std
            self.embedding.weight.data.uniform_(-val, val)

        # Data Properties
        self.normaliser = hparams.normaliser

        self.encoder = Encoder(hparams)
        self.hmm = HMM(hparams)
        self.decoder = FlowSpecDecoder(hparams)
        self.logger = hparams.logger

        print("Use Multilanguage Cathegorical")
        self.emb_l = nn.Embedding(hparams.n_lang, hparams.lin_channels)
        torch.nn.init.xavier_uniform_(self.emb_l.weight)

        print("Use Speaker Embed Linear Norm")
        self.emb_g = nn.Linear(512, hparams.gin_channels)

        print("Use Emo Embed Linear Norm")
        self.emb_emo = nn.Linear(1024, hparams.gin_channels)

    def parse_batch(self, batch):
        """
        Takes batch as an input and returns all the tensor to GPU
        Args:
            batch:

        Returns:

        """
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths, langs, speakers, emos = batch
        text_padded = text_padded.long()
        input_lengths = input_lengths.long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = mel_padded.float()
        gate_padded = gate_padded.float()
        output_lengths = output_lengths.long()
        langs = langs.long()
        speakers = speakers.float()
        emos = emos.float()

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths, langs, speakers, emos),
            (mel_padded, gate_padded),
        )

    def forward(self, inputs):
        text_inputs, text_lengths, mels, max_len, mel_lengths, langs, speakers, emos = inputs
        text_lengths, mel_lengths = text_lengths.data, mel_lengths.data

        l = self.emb_l(langs)
        g = self.emb_g(speakers)
        emo = self.emb_emo(emos)
        
        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
        print(embedded_inputs.shape)
        print(l.transpose(2, 1).expand(embedded_inputs.size(0), embedded_inputs.size(1), -1).shape)
        embedded_inputs = torch.cat((embedded_inputs, l.transpose(2, 1).expand(embedded_inputs.size(0), embedded_inputs.size(1), -1)), dim=-1)
        encoder_outputs, text_lengths = self.encoder(embedded_inputs, text_lengths)

        encoder_outputs = torch.cat([encoder_outputs, g, emo], -1)
        z, z_lengths, logdet = self.decoder(mels, mel_lengths, g=g, emo=emo)
        log_probs = self.hmm(encoder_outputs, text_lengths, z, z_lengths)
        loss = (log_probs + logdet) / (text_lengths.sum() + mel_lengths.sum())
        return loss

    @torch.inference_mode()
    def sample(self, text_inputs, text_lengths=None, langs=None, speakers=None, emos=None, sampling_temp=1.0):
        r"""
        Sampling mel spectrogram based on text inputs
        Args:
            text_inputs (int tensor) : shape ([x]) where x is the phoneme input
            text_lengths (int tensor, Optional):  single value scalar with length of input (x)

        Returns:
            mel_outputs (list): list of len of the output of mel spectrogram
                    each containing n_mel_channels channels
                shape: (len, n_mel_channels)
            states_travelled (list): list of phoneme travelled at each time step t
                shape: (len)
        """
        if text_inputs.ndim > 1:
            text_inputs = text_inputs.squeeze(0)

        if text_lengths is None:
            text_lengths = text_inputs.new_tensor(text_inputs.shape[0])

        l = self.emb_l(langs)
        g = self.emb_g(speakers)
        emo = self.emb_emo(emos)

        text_inputs, text_lengths = text_inputs.unsqueeze(0), text_lengths.unsqueeze(0)
        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
        embedded_inputs = torch.cat((embedded_inputs, l.transpose(2, 1).expand(embedded_inputs.size(0), embedded_inputs.size(1), -1)), dim=-1)

        encoder_outputs, text_lengths = self.encoder(embedded_inputs, text_lengths)
        encoder_outputs = torch.cat([encoder_outputs, g, emo], -1)
        
        (
            mel_latent,
            states_travelled,
            input_parameters,
            output_parameters,
        ) = self.hmm.sample(encoder_outputs, sampling_temp=sampling_temp)

        mel_output, mel_lengths, _ = self.decoder(
            mel_latent.unsqueeze(0).transpose(1, 2), text_lengths.new_tensor([mel_latent.shape[0]]), reverse=True
        )

        if self.normaliser:
            mel_output = self.normaliser.inverse_normalise(mel_output)

        return mel_output.transpose(1, 2), states_travelled, input_parameters, output_parameters

    def store_inverse(self):
        self.decoder.store_inverse()
