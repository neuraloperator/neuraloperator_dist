.. image:: https://img.shields.io/pypi/v/neuraloperator
   :target: https://pypi.org/project/neuraloperator/
   :alt: PyPI

.. image:: https://github.com/NeuralOperator/neuraloperator/actions/workflows/test.yml/badge.svg
   :target: https://github.com/NeuralOperator/neuraloperator/actions/workflows/test.yml


===============
Neural Operator
===============

``neuraloperator`` is a comprehensive library for 
learning neural operators in PyTorch.
It is the official implementation for Fourier Neural Operators 
and Tensorized Neural Operators.

Unlike regular neural networks, neural operators
enable learning mapping between function spaces, and this library
provides all of the tools to do so on your own data.

NeuralOperators are also resolution invariant, 
so your trained operator can be applied on data of any resolution.

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


Installation
------------

Just clone the repository and install locally (in editable mode so changes in the code are immediately reflected without having to reinstall):

prepare

.. code::

  git clone https://github.com/wdlctc/tltorch
  cd torch
  pip install -e .
  cd ..

.. code::

  git clone https://github.com/wdlctc/neuraloperator
  cd neuraloperator
  pip install -e .
  pip install -r requirements.txt

You can also just pip install the library:


.. code::
  
  pip install neuraloperator

Quickstart
----------

After you've installed the library, you can start training operators seemlessly:


.. code-block:: python

   from neuralop.models import FNO

   operator = FNO(n_modes=(16, 16), hidden_channels=64,
                   in_channels=3, out_channels=1)

Tensorization is also provided out of the box: you can improve the previous models
by simply using a Tucker Tensorized FNO with just a few parameters:

.. code-block:: python

   from neuralop.models import TFNO

   operator = TFNO(n_modes=(16, 16), hidden_channels=64,
                   in_channels=3, 
                   out_channels=1,
                   factorization='tucker',
                   implementation='factorized',
                   rank=0.05)

This will use a Tucker factorization of the weights. The forward pass
will be efficient by contracting directly the inputs with the factors
of the decomposition. The Fourier layers will have 5% of the parameters
of an equivalent, dense Fourier Neural Operator!

Checkout the `documentation <https://neuraloperator.github.io/neuraloperator/dev/index.html>`_ for more!

Using with weights and biases
-----------------------------

Create a file in ``neuraloperator/config`` called ``wandb_api_key.txt`` and paste your Weights and Biases API key there.
You can configure the project you want to use and your username in the main yaml configuration files.

Contributing code
-----------------

All contributions are welcome! So if you spot a bug or even a typo or mistake in
the documentation, please report it, and even better, open a Pull-Request on 
`GitHub <https://github.com/neuraloperator/neuraloperator>`_. Before you submit
your changes, you should make sure your code adheres to our style-guide. The
easiest way to do this is with ``black``:

.. code::

   pip install black
   black .

Running the tests
=================

Testing and documentation are an essential part of this package and all
functions come with uni-tests and documentation. The tests are ran using the
pytest package. First install ``pytest``:

.. code::

    pip install pytest
    
Then to run the test, simply run, in the terminal:

.. code::

    pytest -v neuralop
    
Citing
------

If you use NeuralOperator in an academic paper, please cite [1]_, [2]_::

   @misc{li2020fourier,
      title={Fourier Neural Operator for Parametric Partial Differential Equations}, 
      author={Zongyi Li and Nikola Kovachki and Kamyar Azizzadenesheli and Burigede Liu and Kaushik Bhattacharya and Andrew Stuart and Anima Anandkumar},
      year={2020},
      eprint={2010.08895},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
   }

   @article{kovachki2021neural,
      author    = {Nikola B. Kovachki and
                     Zongyi Li and
                     Burigede Liu and
                     Kamyar Azizzadenesheli and
                     Kaushik Bhattacharya and
                     Andrew M. Stuart and
                     Anima Anandkumar},
      title     = {Neural Operator: Learning Maps Between Function Spaces},
      journal   = {CoRR},
      volume    = {abs/2108.08481},
      year      = {2021},
   }


.. [1] Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., and Anandkumar A., “Fourier Neural Operator for Parametric Partial Differential Equations”, ICLR, 2021. doi:10.48550/arXiv.2010.08895.

.. [2] Kovachki, N., Li, Z., Liu, B., Azizzadenesheli, K., Bhattacharya, K., Stuart, A., and Anandkumar A., “Neural Operator: Learning Maps Between Function Spaces”, JMLR, 2021. doi:10.48550/arXiv.2108.08481.
