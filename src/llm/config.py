from pydantic import BaseModel


class GPTConfig(BaseModel):
    batch_size: int = 36
    block_size: int = 1024
    vocab_size: int = 32768
    context_length: int = 1024
    emb_dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    dropout: float = 0.0
    qkv_bias: bool = True


class GPTTrain(BaseModel):
    learning_rate: float = 1e-3
    max_iters: int = 100
    eval_interval: int = 100
    eval_iters: int = 200
