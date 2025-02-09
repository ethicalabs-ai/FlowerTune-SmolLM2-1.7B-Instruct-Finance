# FlowerTune LLM on Finance Dataset

This directory conducts federated instruction tuning with a pretrained [SmolLM2-1.7B-Instruct](HuggingFaceTB/SmolLM2-1.7B-Instruct) model on a [Finance dataset](https://huggingface.co/datasets/FinGPT/fingpt-sentiment-train).
We use [Flower Datasets](https://flower.dev/docs/datasets/) to download, partition and preprocess the dataset.
Flower's Simulation Engine is used to simulate the LLM fine-tuning process in federated way,
which allows users to perform the training on a single GPU.

## Methodology

This baseline performs federated LLM fine-tuning with [DoRA](https://arxiv.org/abs/2402.09353) using the [🤗PEFT](https://huggingface.co/docs/peft/en/index) library.
The clients' models are aggregated with `FedAvg` strategy.
This provides a baseline performance for the leaderboard of Finance challenge.


### SmolLM2-1.7B-Instruct

For the **HuggingFaceTB/SmolLM2-1.7B-Instruct** model I adopted the following fine-tuning methodology:

- **Precision**: `bf16` for model weights.
- **Quantization**: `4-bit` quantization for reduced memory usage.
- **Optimizer**: `paged_adamw_8bit`
- **[DoRA](https://arxiv.org/abs/2402.09353) Configuration**:
  - Rank (r): `32`
  - Alpha: `64`
  - Target Modules:
    - `down_proj`
    - `up_proj`
    - `gate_proj`
- **Training Configuration**:
  - Batch size: `16`
  - Maximum number of steps: `8`
  - Total number of rounds: `12`
  - Fraction fit per round: `0.1`
- **Learning Rate Scheduler**:
  - Cosine Annealing over rounds, where:
    - Maximum LR: `2e-4`
    - Minimum LR: `6e-6`
  - Constant learning rate scheduler over steps
- **Strategy**: `FedAvg`

### Evaluation Results (Accuracy)

- **FiQA**: 56.58 %  
- **FPB**: 71.37 %  
- **TFNS**: 75.76 %  
- **Average**: 67.91 %

### Communication Budget

11005.66 MB

## Environments setup

Project dependencies are defined in `pyproject.toml`. Install them in an activated Python environment with:

```shell
pip install -e .
```

## Experimental setup

The dataset is divided into 50 partitions in an IID fashion, a partition is assigned to each ClientApp.
We randomly sample a fraction (0.1) of the total nodes to participate in each round, for a total of `200` rounds.
All settings are defined in `pyproject.toml`.

> [!IMPORTANT]
> Please note that `[tool.flwr.app.config.static]` and `options.num-supernodes` under `[tool.flwr.federations.local-simulation]` are not allowed to be modified for fair competition if you plan to participated in the [LLM leaderboard](https://flower.ai/benchmarks/llm-leaderboard).


## Running the challenge

Run the challenge with default config values.

The configs are defined in `[tool.flwr.app.config]` entry of `pyproject.toml`, and are loaded automatically.

```bash
flwr run
```

## Running the evaluation

Please check [flowertune-eval-finance](./flowertune-eval-finance).


## Model saving

The global PEFT model checkpoints are saved every 5 rounds after aggregation on the sever side as default, which can be specified with `train.save-every-round` under [tool.flwr.app.config] entry in `pyproject.toml`.
