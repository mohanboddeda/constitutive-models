# Synthetic Data Generation for solving the Maxwell-B Equation

## Constitutive Models

This repository contains the code and data pipelines for my Master's thesis, which focuses on applying machine learning to non-Newtonian fluid mechanics. The project utilizes JAX to build, train, and evaluate physics-informed neural networks and constitutive models for various complex fluids.

## Project Structure

The repository is organized into several key directories and scripts to keep the workflow modular and clean:

### Directories
* **`config/`**: Contains YAML configuration files for data generation, model architecture, and training setups for models like Maxwell-B.
* **`datafiles/`**: Stores the generated datasets used for training and evaluating the machine learning models.
* **`diagnostic_plots/` & `images/`**: Contains visualizations, stagewise trends, and performance plots generated during data generation and preprocessing.
* **`trained_models/`**: Holds the saved weights and parameters of the fully trained models for all 3 methods: Random, Uniform, and Flow models.
* **`utils/`**: Includes essential helper scripts, such as `invariants.py` for calculating tensor invariants. All util files are called frequently inside training scripts.

### Core Data Generation Scripts
These scripts are responsible for creating the specific fluid datasets required for machine learning:
* **`generateRandomdata.py`**: A pipeline for generating randomized data distributions for the Maxwell model.
* **`generateUniformMaxwell.py`**: Generates uniformly distributed data specifically for the Maxwell model.
* **`generateFlowMaxwell_Mining.py`**: Extracts and structures specific fluid flow data for the Maxwell model.

### Core Machine Learning Scripts (JAX)
These scripts handle the tensor-based neural network training using the JAX framework:
* **`TensorJAX_Random.py`**: Trains the models using the randomized datasets.
* **`TensorJAX_UniformNet2Net.py`**: A training pipeline tailored for uniformly distributed data.
* **`TensorJAX_Flownet2net.py`**: Trains the network using structured flow data.

## Constitutive Models Implemented
The codebase supports several non-Newtonian fluid models, including:
* Maxwell (e.g., Maxwell-B)

## Getting Started & Installation

To run the codes in this repository, you will need to install the required dependencies. It is highly recommended to do this inside a Python virtual environment.

1. Clone the repository to your local machine and navigate into the folder:
```bash
git clone [https://github.com/mohanboddeda/constitutive-models.git](https://github.com/mohanboddeda/constitutive-models.git)
cd constitutive-models

## Acknowledgements and Foundational Work

This research builds upon the extraordinary contributions of several authors in the fields of Physics-Informed Neural Networks (PINNs), complex fluid rheology, and transfer learning methodologies. While not exhaustive, this work is particularly indebted to the foundational concepts established in the following papers:

* **Physics-Informed Neural Networks:** Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics, 378*, 686–707.
* **Rheology-Informed Neural Networks:** Mahmoudabadbozchelou, M., Jamali, S., & Kamani, K. (2021). Rheology-informed neural networks (RhINNs) for the characterization of complex fluids. *Scientific Reports, 11*(1), 1–13.
* **Net2Net Transfer Learning:** Chen, T., Goodfellow, I., & Shlens, J. (2016). Net2Net: Accelerating Learning via Knowledge Transfer. *International Conference on Learning Representations (ICLR)*.
* **Accelerated Training & Transfer Learning:** * Yuan, X., Savarese, P., & Maire, M. (2023). Accelerated Training via Incrementally Growing Neural Networks using Variance Transfer and Learning Rate Adaptation. *Advances in Neural Information Processing Systems*.
  * Bahmani, B., & Sun, W. (2021). Training multi-objective/multi-task collocation physics-informed neural network with student/teachers transfer learnings. *arXiv preprint arXiv:2107.11496*.
  * Monaco, S., & Apiletti, D. (2023). Training physics-informed neural networks: One learning to rule them all? *Results in Engineering, 18*, 101023.
  * Liu, Y., et al. (2023). Adaptive transfer learning for physics-informed neural networks. *arXiv*.
  * Wang, Y., et al. (2024). Transfer Learning in Physics-Informed Neural Networks: Full Fine-Tuning, Lightweight Fine-Tuning, and Low-Rank Adaptation. *arXiv preprint 2502.00782*.
* **Curriculum Learning:** Bengio, Y., et al. (2009). Curriculum learning. *Proceedings of the 26th Annual International Conference on Machine Learning (ICML)*, 41–48.
* **JAX Framework:** Bradbury, J., et al. (2018). *JAX: Composable transformations of Python+NumPy programs* (Version 0.3.13) [Software]. http://github.com/google/jax
## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check the [issues page](https://github.com/mohanboddeda/constitutive-models/issues) if you have any questions or want to suggest improvements. 

## Citation

If you use this code or dataset in your academic research, please cite this repository:

```bibtex
@software{boddeda_constitutive_models_2026,
  author = {Boddeda, Mohan},
  title = {Synthetic Data Generation for solving the Maxwell-B Equation},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{[https://github.com/mohanboddeda/constitutive-models](https://github.com/mohanboddeda/constitutive-models)}}
}
