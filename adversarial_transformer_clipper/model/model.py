from typing import Optional
import torch.nn as nn, torch
import math

from ..config import EncoderGeneratorConfig
from .audio_features import ResNetAudio


class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [Batch, Sequence, Features]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class EncoderGeneratorModel(nn.Module):
    def __init__(self, args: EncoderGeneratorConfig) -> None:
        super().__init__()

        self.audio_feature_extractor = ResNetAudio(
            layers=args.audio_feature_extractor.layers, #[2, 2, 2, 2]
            output_depth=args.d_model, # 1024
            input_filter=args.audio_feature_extractor.input_filter, # 1
        )
        self.video_feature_extractor = None # TODO Experiment with video in the future

        self.positional_encoder = PositionalEncoding(
            d_model=args.d_model,
            dropout=args.dropout
        )
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=args.d_model,
            nhead=args.nhead,
            d_hid=args.d_hid,
            dropout=args.dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers=encoder_layers,
            nlayers=args.nlayers
        )
    
    def forward(self, audio_input: torch.Tensor, video_input: Optional[torch.Tensor]=None, mask_input: Optional[torch.Tensor]=None):
        """
        Args:
            audio_input: Tensor, shape [Batch, Sequence, Channel, Height, Width]
            video_input: Tensor, shape [Batch, Sequence, Channel, Height, Width], Optional
            mask_input: Tensor, shape [Batch, Sequence, Sequence], Optional

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        # Extract audio features | (B, S, C, H, W) -> (B*S, C, H, W) -> (B*S, E) -> (B, S, E)
        audio_features = self.audio_feature_extractor(audio_input.view(-1, *list(audio_input.shape[2:])))
        audio_features = audio_features.view(*list(audio_input.shape[:2]), -1)

         # Extract video features and concatenate with audio
        if self.video_feature_extractor is not None:
            video_features = self.video_feature_extractor(video_input)
            features = torch.cat((audio_features, video_features), dim=-1)
        else:
            features = audio_features

            # video_features = torch.empty_like(audio_features, layout=list(audio_features.shape[:-1])+[0])
        # features = torch.cat((audio_features, video_features), dim=-1)

        # TODO Add padding mechanism (even though it will certainly be useless)

        # Encode position
        features = self.positional_encoder(features)

        # Encode sequences
        output = self.transformer_encoder(features, mask_input)

        return output


class QueryGeneratorModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, latent_vector):
        pass


class DecoderGeneratorModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, key, query):
        pass


class GeneratorModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.encoder_module = EncoderGeneratorModel()
        self.query_module = QueryGeneratorModel()
        self.decoder_module = DecoderGeneratorModel()

    def forward(self, audio_input, video_input, latent_vector):
        """
        |https://stats.stackexchange.com/a/424127
        |   The key/value/query concept is analogous to retrieval systems.
        |   For example, when you search for videos on Youtube, the search engine will map your query (text in the search bar)
        |   against a set of keys (video title, description, etc.) associated with candidate videos in their database,
        |   then present you the best matched videos (values).
        |   The attention operation can be thought of as a retrieval process as well.


        Args:
            audio_input (_type_): Audio wave (or MEL spectrogram?) of a podcast/video
            video_input (_type_): Frames of a podcast/video (optionnal?)
            latent_vector (_type_): Noise for now, maybe text query in the future

        Returns:
            _type_: Set of segments (start/end timestamps?) to concatenate to create a clip from the audio/video input
        """
        key = self.encoder_module(audio_input, video_input)
        query = self.query_module(latent_vector)
        value = self.decoder_module(key, query)
        return value


class DiscriminatorModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, audio_input, video_input):
        """

        Args:
            audio_input (_type_): Audio wave (or MEL spectrogram?) of a clip
            video_input (_type_): Frames of a clip (optionnal?)
        
        Returns:
            _type_: Discimination score between generated or groundtruth clip
        """
        pass