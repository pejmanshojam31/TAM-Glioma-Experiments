# Glioma-Macrophage-Oxygen Simulation

## Overview

This project implements a numerical simulation of glioma-macrophage-oxygen interactions using a reaction-diffusion partial differential equation (PDE) model. The simulation models how glioma cells, macrophages, and oxygen diffuse and interact over time within a one-dimensional spatial domain. It employs finite difference methods (FDM) for spatial discretization and an implicit BDF solver for time integration, ensuring numerical stability for stiff systems.

## Features

Solves a reaction-diffusion PDE system modeling glioma cells (C), macrophages (M), and oxygen (n).

Implements nonlinear diffusion coefficients that depend on oxygen and cell densities.

Uses logistic growth terms to model glioma and macrophage proliferation.

Employs a sigmoidal oxygen modulation function for regulatory effects.

Applies Neumann boundary conditions (zero-flux) at both spatial boundaries.

Utilizes an interactive GUI with PyQt5 to adjust model parameters and visualize results.

## Numerical Methods

Finite Difference Method (FDM): Used for spatial discretization with central differences.

Implicit Time Integration (BDF): solve_ivp(method='BDF') is used for stable time evolution of stiff PDEs.

Neumann Boundary Conditions: Ensures no-flux conditions at domain edges.

## Installation

To run the simulation, ensure you have the following dependencies installed:
'''
pip install numpy scipy matplotlib pyqt5
'''

Running the Simulation

Run the following command:

'''
python main.py
'''


This will launch the GUI where you can adjust parameters and visualize simulation results.

GUI Features

Adjustable sliders to modify model parameters.

A Run Simulation button to compute and display results.

Matplotlib plots showing the final spatial distributions of glioma cells, macrophages, and oxygen.

File Structure

main.py: Main script containing the GUI and numerical solver.

pde_solver.py: Contains the PDE model and numerical integration functions.

gui.py: Implements the PyQt5-based graphical user interface.

License

This project is released under the MIT License.
