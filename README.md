# Deep Mixture of Experts (MoE) Reproduction

This repository contains the reproduction of the experiments presented in the paper [**"Learning Factored Representations in a Deep Mixture of Experts"**](https://arxiv.org/abs/1312.4314) by David Eigen, Marc'Aurelio Ranzato, and Ilya Sutskever.

## Paper Details

**BibTeX Citation:**
```bibtex
@article{eigen2013learning,
  title={Learning factored representations in a deep mixture of experts},
  author={Eigen, David and Ranzato, Marc'Aurelio and Sutskever, Ilya},
  journal={arXiv preprint arXiv:1312.4314},
  year={2013}
}
```

## Repository Structure
- `MoE.py`: This script contains the implementation of the Deep Mixture of Experts (deepMoE) model as described in the paper.
- `mnist.py`: This script uses the MNIST dataset to train the deepMoE model and visualize the results.
- `model100_20_new.pth`: Pre-trained model parameters obtained after training the deepMoE model.
- `layer1_class.png`, `layer1_translation.png`, `layer2_class.png`, `layer2_translation.png`: These are the visualization results from training the model on the MNIST dataset.

### Usage

1. **Training the Model:**

   To train the deepMoE model on the MNIST dataset, run:
   ```bash
   python mnist.py

2. **Visualizing the Results:**

   The visualizations from training can be found in the four PNG files provided in the repository. These files demonstrate the performance and representation learning of the deepMoE model on the MNIST dataset.
