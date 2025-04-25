from dataclasses import dataclass, field

@dataclass
class Run:
    fnn_size: int
    lstm_size: int
    num_samples: int
    model: any = None
    num_params: int = 0
    training_data: any = None
    training_losses: any = field(default_factory=list)