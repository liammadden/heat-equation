from dataclasses import dataclass, field

@dataclass
class Run:
    m: int
    model: any = None
    num_params: int = 0
    training_losses: any = field(default_factory=list)
    test_losses: any = field(default_factory=list)