from dataclasses import dataclass, asdict, field
from typing import Optional

@dataclass
class AudioFrameLoaderConfig:
    frame_rate: float = 1.0
    sampling_rate: int = 44_100
    hop_length: int = 441
    win_length: int = 2024
    n_mels: int = 128
    n_fft: int = 4096
    f_max: Optional[int] = None
    hop_duration: float = field(init=False)
    wave_frame_length: int = field(init=False)

    def __post_init__(self):
        self.hop_duration = self.hop_length/self.sampling_rate
        self.wave_frame_length = int(self.sampling_rate//self.frame_rate)

@dataclass
class EncoderGeneratorConfig:
    test: int