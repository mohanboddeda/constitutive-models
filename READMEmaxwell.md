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

How to Run the Code (Examples)
This project uses a modular approach, where data is generated first and then fed into the respective JAX training pipelines. You can easily modify hyperparameters directly from the command line.

Below are example commands for running the different pipelines available in this repository:

1. Randomized Data Pipeline
First, generate the multi-stage or single-stage random data with custom boundaries:

Bash
python generateRandomdata.py mode=multi_stage n_samples=10000 +custom_min=1.2 +custom_max=1.4
Next, train the model using the generated random data and a pre-trained checkpoint:

Bash
python TensorJAX_Random.py mode=multi_stage stage_tag="1.4_1.6" transfer_checkpoint="./trained_models/random/multi_stage/seed_42/20ksamples/maxwell_B_1.2_1.4/best_checkpoint.msgpack" training.batch_size=128 training.learning_rate=1e-4 +n_samples=10000

2. Uniform Data Pipeline
Generate the uniformly distributed Maxwell data:

Bash
python generateUniformMaxwell.py mode=multi_stage n_samples=10000 seed=42
Train the Net2Net architecture on the uniform data, defining the specific layers and epochs:

Bash
python TensorJAX_UniformNet2Net.py mode=multi_stage +stage="1.0_1.8" +n_samples=10000 model.layers="[9,128,128,128,6]" +training.batch_size=32 +training.learning_rate=1e-4 +training.num_epochs=500 transfer_checkpoint=null config_name="uniform_net2net_config"

3. Flow Data (Mining) Pipeline
Generate specific flow data, such as biaxial extension, defining the rates and boundaries:

Bash
python generateFlowMaxwell_Mining.py mode=multi_stage flow_types="['biaxial_extension']" ++n_samples=10000 ++custom_min=1.0 ++custom_max=1.2 ++rate_min=0.0 ++rate_max=2.0
Train the model on the generated flow data while enforcing physics constraints (lambda_phys):

Bash
python TensorJAX_Flownet2net.py --config-name flow_net2net_config mode=multi_stage flow_types="['biaxial_extension']" ++stage="1.0_1.2" ++n_samples=3000 transfer_checkpoint=null ++model.layers="[9, 128, 128, 128, 6]" ++training.batch_size=64 ++training.learning_rate=1e-4 ++training.weight_decay=1e-4 ++training.num_epochs=1500 ++training.lambda_phys=0.3