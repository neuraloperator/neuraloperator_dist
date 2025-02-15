[![PyPI](https://img.shields.io/pypi/v/neuraloperator)](https://pypi.org/project/neuraloperator/)
[![Tests](https://github.com/NeuralOperator/neuraloperator/actions/workflows/test.yml/badge.svg)](https://github.com/NeuralOperator/neuraloperator/actions/workflows/test.yml)

# Neural Operator

**neuraloperator** is a comprehensive library for learning neural operators in PyTorch.
It is the official implementation for **Fourier Neural Operators (FNO)** and 
**Tensorized Neural Operators (TFNO)**.

Unlike regular neural networks, neural operators enable learning mappings 
between **function spaces**, providing a resolution-invariant framework 
for tasks such as modeling partial differential equations (PDEs). 
This library gives you all the necessary tools to build and train these 
operators on your own datasets.

---

## New Feature: Image Parallelism (IP)

We now include an **image parallelism** module, inspired by the 
[DeepSpeed Ulysses](https://github.com/microsoft/DeepSpeedExamples/tree/master/NewsQA) approach, 
which enables large-scale distributed training for PDE-based tasks. 

### How it works

1. **All-to-All Communication**: We perform **two all-to-all** operations around the FNO block 
   (before and after) to distribute and gather image “patches” (or PDE field snapshots) 
   across multiple GPUs. 
2. **Scaling with Number of GPUs**: This design effectively partitions large images across 
   available GPUs, allowing you to train on significantly larger image (or field) sizes 
   than would fit on a single GPU.

You can view the detailed implementation in 
[`neuralop/layers/fno_block.py`](https://github.com/neuraloperator/neuraloperator_dist/blob/main/neuralop/layers/fno_block.py).

A convenient **one-click run** script, `python neural_ip.py`, is provided to demonstrate 
image-parallel training on a sample PDE dataset. Give it a try to see how easily 
you can scale up your NeuralOperator workflows!

---

## Installation

Clone the repository and install locally (in editable mode so that changes in 
the code are immediately reflected without having to reinstall):

### 1. Prepare `tltorch` (for tensorization features)

```bash
git clone https://github.com/wdlctc/tltorch
cd tltorch
pip install -e .
cd ..
```

### 2. Install `neuraloperator`

```bash
git clone https://github.com/wdlctc/neuraloperator
cd neuraloperator
pip install -e .
pip install -r requirements.txt
```

Alternatively, you can install via pip directly:

```bash
pip install neuraloperator
```

---

## Quickstart

After installation, you can start training operators seamlessly.

### Fourier Neural Operator (FNO)

```python
from neuralop.models import FNO

operator = FNO(
    n_modes=(16, 16),
    hidden_channels=64,
    in_channels=3,
    out_channels=1
)
```

### Tensorized FNO (TFNO)

Using a **Tucker** factorization (just as an example) to drastically reduce parameters:

```python
from neuralop.models import TFNO

operator = TFNO(
    n_modes=(16, 16),
    hidden_channels=64,
    in_channels=3,
    out_channels=1,
    factorization='tucker',
    implementation='factorized',
    rank=0.05
)
```

By using a **Tucker** factorization, the model’s weight matrices are decomposed, 
resulting in significantly fewer parameters while maintaining comparable performance. 

For comprehensive examples, check out the 
[documentation](https://neuraloperator.github.io/neuraloperator/dev/index.html).

---

## Running with Image Parallelism

To run the **image parallelism** demo (inspired by **DeepSpeed Ulysses**), simply execute:

```bash
python neural_ip.py
```

This script demonstrates how to distribute PDE field snapshots (or "images") 
across multiple GPUs, enabling large-scale training and memory savings.

---

## Using with Weights & Biases

Create a file named `wandb_api_key.txt` in `neuraloperator/config` and paste 
your Weights & Biases API key inside. You can customize your project name and username 
in the main YAML configuration files.

---

## Contributing Code

All contributions are welcome! If you spot a bug, typo, or any issue, 
please report it or open a Pull Request on our 
[GitHub](https://github.com/NeuralOperator/neuraloperator).

Before submitting changes, ensure your code adheres to our style guide. 
The easiest way is with [`black`](https://github.com/psf/black):

```bash
pip install black
black .
```

---

## Running the Tests

We use [`pytest`](https://docs.pytest.org/en/stable/) for testing. To run tests:

```bash
pip install pytest
pytest -v neuralop
```

---

## Citing

If you use **NeuralOperator** in an academic paper, please cite the following:

**[1]** Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., 
Stuart, A., and Anandkumar, A. *Fourier Neural Operator for Parametric 
Partial Differential Equations*, ICLR, 2021.  
[arXiv:2010.08895](https://arxiv.org/abs/2010.08895)

**[2]** Kovachki, N., Li, Z., Liu, B., Azizzadenesheli, K., Bhattacharya, K., 
Stuart, A., and Anandkumar, A. *Neural Operator: Learning Maps Between 
Function Spaces*, JMLR, 2021.  
[arXiv:2108.08481](https://arxiv.org/abs/2108.08481)
