# Constitutive Modeling for the Navier-Stokes Equations

## 🚀 Installation des Pytonenvironments

Um dieses Projekt auszuführen, benötigst du eine **Conda**-Installation (Anaconda oder Miniconda). Folge diesen Schritten, um das Projekt-Environment einzurichten.

### 1\. Repository klonen

Klone zuerst das Repository auf deinen lokalen Rechner und wechsle in das Verzeichnis:

```bash
git clone https://github.com/kiranboddeda9/constitutive-models.git
cd constitutive-models
```

### 2\. Conda-Umgebung erstellen

Nutze die mitgelieferte `environment.yml`-Datei, um die Conda-Umgebung mit allen benötigten Paketen automatisch zu erstellen.

```bash
conda env create -f environment.yml
```

Dieser Befehl erstellt eine neue Umgebung mit dem Namen `rheoML` und installiert alle Abhängigkeiten.

### 3\. Umgebung aktivieren

Aktiviere die neu erstellte Umgebung, um sie zu nutzen:

```bash
conda activate rheoML
```

### 4\. Code ausführen

Jetzt ist alles bereit\! Du kannst nun die Skripte des Projekts ausführen. Zum Beispiel:

```bash
python generateMaxwell.py dim=3
      oder
python train_model.py 
```

-----


---
## Problem Formulation:

This project focuses on addressing the **closure problem** inherent in the Navier-Stokes equations for fluid dynamics. We explore the modeling of the stress tensor, which is essential for solving the system of equations, contrasting the simplicity of Newtonian fluids with the complexity of non-Newtonian models.

---
## Governing Equations

The motion of a fluid is described by the principles of conservation of mass and momentum. In their differential form, these are known as the Navier-Stokes equations.

The **conservation of mass** (or continuity equation) is given by:
$$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{u}) = 0$$
where $\rho$ is the fluid density, $t$ is time, and $\mathbf{u}$ is the fluid velocity vector.

The **conservation of momentum** is expressed as:
$$\frac{\partial (\rho \mathbf{u})}{\partial t} + \nabla \cdot (\rho \mathbf{u} \otimes \mathbf{u}) = -\nabla p + \nabla \cdot \mathbf{T} + \rho \mathbf{g}$$
Here, $p$ is the pressure, $\mathbf{T}$ is the deviatoric stress tensor, and $\mathbf{g}$ is the acceleration due to gravity. The term $\mathbf{u} \otimes \mathbf{u}$ represents the outer product of the velocity vector with itself.

The **closure problem** arises because the stress tensor $\mathbf{T}$ is an unknown variable. To solve this system, we need an additional equation that relates $\mathbf{T}$ to the other flow variables. This is known as a **constitutive equation**.

---
## Newtonian Fluids

For Newtonian fluids, the relationship between stress and the rate of strain is linear and straightforward. The constitutive equation, also known as Newton's law of viscosity, closes the system of equations.

The stress tensor $\mathbf{T}$ is defined as:
$$\mathbf{T} = 2\eta \mathbf{D}$$
where $\eta$ is the dynamic viscosity and $\mathbf{D}$ is the rate of strain tensor, given by:
$$\mathbf{D} = \frac{1}{2} \left( \nabla \mathbf{u} + (\nabla \mathbf{u})^T \right)$$
By substituting this expression for $\mathbf{T}$ into the momentum equation, the system is closed and can be solved.

---
## Non-Newtonian Fluids

For non-Newtonian fluids, the relationship between stress and strain rate is more complex and often non-linear. Simple algebraic models are insufficient, and a new differential equation is required to describe the evolution of the stress tensor. This adds another layer of complexity to the closure problem.

A general form for a **differential constitutive model** can be expressed as:
$$\lambda_1 \frac{\mathcal{\delta_a} \mathbf{T}}{\mathcal{\delta} t} + \mathbf{T} = 2\eta_0 \left( \mathbf{D} + \lambda_2 \frac{\mathcal{D} \mathbf{D}}{\mathcal{D} t} \right)$$
Here, we introduce four new parameters:
* $\eta_0$: The zero-shear viscosity.
* $\lambda_1$: The relaxation time, which describes how long it takes for stress to dissipate.
* $\lambda_2$: The retardation time, which describes the delayed strain response.
* $\frac{\mathcal{\delta_a}}{\mathcal{\delta}t}$: An objective time derivative (e.g., the Upper-Convected time Derivative) that makes the model independent of the observer's reference frame.

This additional differential equation for the stress tensor $\mathbf{T}$ must be solved simultaneously with the mass and momentum equations, significantly increasing the computational challenge.

---

## Project Workflow

This project follows a systematic workflow from theoretical modeling to practical implementation and validation in a CFD environment. The core idea is to replace a traditional, hard-coded constitutive model with a trained Physics-Informed Neural Network (PINN) inside OpenFOAM.

### 1. Model Specialization: Maxwell-B Model

We start with the **Generalized Differential Constitutive Model**:
$$\lambda_1 \frac{\mathcal{\delta_a} \mathbf{T}}{\mathcal{\delta} t} + \mathbf{T} = 2\eta_0 \left( \mathbf{D} + \lambda_2 \frac{\mathcal{D} \mathbf{D}}{\mathcal{D} t} \right)$$To obtain the simpler **Maxwell-B model**, we set the retardation time parameter $\lambda_2 = 0$. This model serves as our initial proof of concept.$$\lambda_1 (\frac{\mathcal{d} \mathbf{T}}{\mathcal{d} t} - \mathbf{L}\mathbf{T} - \mathbf{L}\mathbf{T}^T) + \mathbf{T} = 2\eta_0 \mathbf{D}$$

---
### 2. Synthetic Data Generation

A high-quality dataset will be generated synthetically by numerically solving the Maxwell-B equation.

* **Input Data:** The primary input for the model is the **velocity gradient tensor, $\mathbf{L}$**.
* **Output Data:** A numerical solver will compute the resulting stress tensor response, $\mathbf{T}(t)$, for various input histories of $\mathbf{L}(t)$.

This process yields a labeled dataset $(\mathbf{L}(t), \mathbf{T}(t))$ that represents the "ground truth" for training our network.

---
### 3. PINN Training and Validation 🧠

The generated dataset will be used to train a **Physics-Informed Neural Network (PINN)**.

* **Training:** The PINN will learn the mapping from an input history of $\mathbf{L}$ to the resulting stress $\mathbf{T}$.
* **Validation:** After training, the PINN's performance will be rigorously validated against a separate, unseen test dataset to ensure it has accurately learned the constitutive relationship.

---
### 4. Integration into OpenFOAM 🚀

The fully trained and validated PINN will be deployed as a functional constitutive model within **OpenFOAM**. At each time step, the solver will pass the local velocity gradient $\mathbf{L}$ to the PINN and receive the predicted stress tensor $\mathbf{T}$ back to solve the momentum equation.

---
### 5. Final Validation with OpenFOAM ✅

The final step is to validate the entire hybrid solver by comparing its results against a conventional OpenFOAM simulation that uses the hard-coded mathematical Maxwell-B equation for a benchmark flow problem.

---
## Further Considerations and Research Questions

This project opens up several advanced topics and challenges that need to be addressed.

### Model Expansion
While the Maxwell-B model is a good starting point, more complex and realistic models should be considered:
* **Oldroyd-B Model:** Incorporates a solvent viscosity, making it suitable for polymer solutions.
* **Giesekus or Phan-Thien-Tanner (PTT) Models:** These non-linear models can capture important physical phenomena like shear-thinning, which the Maxwell model cannot.

### Generalization and Dimensionless Formulation
A key research question is whether a single PINN can be trained to represent multiple fluids.
* **Dimensionless Equations:** By recasting the constitutive equation into a dimensionless form using the material parameters ($\lambda_1, \eta_0$), we can create dimensionless groups.
* **Generalized PINN:** A PINN could be trained on this universal, dimensionless relationship. To model a specific fluid, one would simply provide its material parameters to scale the inputs and outputs of the already trained, generalized PINN. This would make the model highly versatile.

### The Problem of Time Dependence
Constitutive models are differential equations in time, meaning the current stress depends on the history of deformation. A simple feed-forward neural network cannot capture this "memory". We must consider architectures designed for temporal data:
* **Model:** In the file lightning.py is a Transformer model that is able to predict time series.
* **PINN Loss Function:** The time derivative term $\frac{\mathcal{D}\mathbf{T}}{\mathcal{D}t}$ can be incorporated into the loss function using automatic differentiation, forcing the network to learn the time-dependent behavior.

---
## Summary of Key Challenges & Questions

* **Model Selection:** Which constitutive model beyond Maxwell-B (Oldroyd-B, Giesekus) offers the best balance of accuracy and trainability for the PINN?
* **Temporal Dependency:** What is the best technique to correctly capture the fluid's memory effects?
* **Generalization:** How can we formulate and train a single, dimensionless PINN that is independent of specific material parameters but can be adapted to any fluid?
* **PINN Architecture:** How do we design the network (depth, width, activation functions) to ensure it is both efficient for OpenFOAM and accurate?
* **Software Integration:** What is the most robust and computationally efficient way to interface a Python-trained model (e.g., PyTorch) with the C++ environment of OpenFOAM?
* **Validation Strategy:** What benchmark problems are best suited to validate the PINN-based solver against traditional methods and highlight its advantages or limitations?
