from turtle import forward
from typing import Callable, Optional
import torch.nn as nn, torch


class EncoderGeneratorModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # self.audio_feature_extractor = 
    def forward(self, audio_input, video_input):
        pass


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