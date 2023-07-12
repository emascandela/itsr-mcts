Image Transformation Sequence Retrieval with a General Reinforcement Learning Algorithm
==================

Code for paper **Image Transformation Sequence Retrieval with a General Reinforcement Learning Algorithm**. It includes the training and evaluation scripts implemented with TensorFlow, as well as the code needed for generating the datasets used in our work, which can be used to compare with other approaches.

# Project structure

- `configs/` Includes the experiment configuration. Each file contains the configuration of each model trained in our work.
- `data/` Directory containing the data needed for generating the datasets.
- `generators/` An API for generating the dataset samples.
- `models/` The code for building the models trained in the paper.
- `results/` Folder for storing the models' weights.
- `train.py` Training script
- `evaluate.py` Evaluation script

# Data generation and reproducibility
In this project, we include the code needed for reproducing the results of the paper as well as the datasets used and the code for generating the training samples. For each of the scenarios, we provide the code necessary for generating the same partitions that are considered in the paper.

The `generators.Generator` class is an interface for implementing different generations for the ITSR problem. The generator class includes We provide implementations for both scenarios: the toy scenario with the free (`generators.GridComposer`) and the constrained (`generators.SequenceGridComposer`) setups and the real scenario (`generators.ImagenetteProecssor`).

For instantiating a `Generator` implementation, we can use the factory method `generators.get_generator` which constructs and returns a generator for each of the scenarios described in the paper. In this function, we have to specify the scenario we want to generate and the data split we want to use. The available scenarios are:
- `toy_free` for the toy scenario with the free setup.
- `toy_constrained` for the toy scenario with the constrained setup.
- `real` for the real scenario.

The available splits are: `train`, `val`, and `test` which provides the same generator in the toy scenario. In the real scenario, 8 of the Imagenette classes are used in the training split, 1 in the validation split, and 1 in the testing split. **For reproducibility purposes** we use a seed for the random generator. This seed is defined as a default argument in the generator factory and can be modified with the `seed` parameter.


Usage example:
```python
from generators import get_generator

real_scenario_generator = get_generator(scenario="real", split="test")
pair = real_scenario_generator.get_random_pair()

source_image = pair.source_image.numpy()
target_image = pair.target_image.numpy()

gt_transformations = pair.target_image.applied_transformations()
```

This will return the source and target images in `np.ndarray` format in `source_image` and `target_image` variables and the transformation sequence applied to the source image for generating the target image.

## Imagenette download
For producing the real scenario data we have to download the Imagenette dataset:

```bash
mkdir data && cd data
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
tar xvf imagenette2.tgz
```

We expect the directory structure to be the following:
```
project_dir/
  data/
    imagenette2/
      train/
      val/
      noisy_imagenette.csv
    
```

# Requirements
This is tested in Ubuntu 22.04 with Python 3.10.8.

We can install the python requirements with pip:
```bash
pip install -r requirements
```

# Running the experiments
The training and evaluating scripts are in `train.py` and `evaluate.py` files, respectively. We can run them specifying the name of the model configuration we want to train or evaluate. The model configurations are defined in the `config` dir. The training script will train the model and store its weights in the `results` directory. The evaluation script will load the trained model stored in the `results` directory, evaluate the model and print the metrics for the single-shot and the top-k evaluations.

Usage:
```python
python train.py EXPERIMENT_NAME
python evaluate.py EXPERIMENT_NAME
```
Example:
```python
python train.py real_convnext_mcts
python evaluate.py real_convnext_mcts
```