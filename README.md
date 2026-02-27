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
