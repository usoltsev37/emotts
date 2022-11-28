from dataclasses import dataclass, field
from typing import List, Literal



@dataclass
class DecoderParams:
    conv_kernel_size: List[int] = field(default_factory=lambda: [9, 1])
    conv_filter_size: int = field(default=1024)
    decoder_layer: int = field(default=6)
    decoder_head: int = field(default=2)
    decoder_hidden: int = field(default=256)
    decoder_dropout: float = field(default=0.2)



@dataclass
class EncoderParams:
    conv_kernel_size: List[int] = field(default_factory=lambda: [9, 1])
    conv_filter_size: int = field(default=1024)
    encoder_layer: int = field(default=6)
    encoder_head: int = field(default=2)
    encoder_hidden: int = field(default=256)
    encoder_dropout: float = field(default=0.2)




@dataclass
class VariancePredictorParams:
    filter_size: int = field(default=256)
    kernel_size: int = field(default=3)
    dropout: float = field(default=0.5)      

@dataclass
class VarianceEmbeddingParams:
    n_bins: int = field(default=256)
    pitch_quantization: Literal['linear', 'log'] = 'linear'
    energy_quantization: Literal['linear', 'log'] = 'linear' # support 'linear' or 'log', 'log' is allowed only if the energy values are not normalized during preprocessing
    

@dataclass
class VarianceAdaptorParams:
    predictor_params: VariancePredictorParams
    embedding_params: VarianceEmbeddingParams



@dataclass
class FastSpeech2Params:

    encoder_params: EncoderParams
    decoder_params: DecoderParams
    max_seq_len: int = field(default=1000)
    use_gst: bool = field(default=False)





