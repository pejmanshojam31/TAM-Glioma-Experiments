import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Use the Qt5Agg backend for embedding in PyQt5
matplotlib.use("Qt5Agg")

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QScrollArea
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.integrate import solve_ivp

###############################################################################
# PDE model functions
###############################################################################
def derivative(v, dx):
    """Compute first derivative using central differences with Neumann BC."""
    dv = np.zeros_like(v)
    dv[1:-1] = (v[2:] - v[:-2]) / (2*dx)
    dv[0]    = (v[1] - v[0]) / dx
    dv[-1]   = (v[-1] - v[-2]) / dx
    return dv

def pde_rhs(t, U, params, xh, Nx):
    U_reshaped = U.reshape(3, Nx)
    C, M, n = U_reshaped[0, :], U_reshaped[1, :], U_reshaped[2, :]

    # Sigmoidal modulation for oxygen using fixed theta = 20 and n_cr = 0.25
    sigma_n = 1 - 1/(1 + np.exp(-2 * params['theta'] * (n - params['n_cr'])))

    # Modified diffusion coefficients
    D_C_mod = (params['D_C'] * (1 + params['gamma_h'] * sigma_n) +
               params['D_C'] * (params['gamma_n'] * (M / params['K'])) -
               params['D_C'] * (params['gamma_h_hat'] * sigma_n * (M / params['K'])))
    D_M_mod = (params['D_m'] * (1 + params['alpha_h'] * sigma_n) +
               params['D_m'] * (params['alpha_n'] * (C / params['K'])) +
               params['D_m'] * (params['alpha_h_hat'] * sigma_n * (C / params['K'])))
    D_n_const = params['D_n']

    # Spatial derivatives and fluxes
    dCdx = derivative(C, xh)
    dMdx = derivative(M, xh)
    dndx = derivative(n, xh)
    flux_C = D_C_mod * dCdx
    flux_M = D_M_mod * dMdx
    flux_n = D_n_const * dndx
    div_flux_C = derivative(flux_C, xh)
    div_flux_M = derivative(flux_M, xh)
    div_flux_n = derivative(flux_n, xh)

    # Reaction terms
    r_c_mod = (params['r_c'] * (1 - params['delta_h'] * sigma_n) -
               params['r_c'] * (params['delta_n'] * (M / params['K_m'])) +
               params['r_c'] * (params['delta_h_hat'] * sigma_n * (M / params['K_m'])))
    reaction_C = r_c_mod * C * (1 - C / params['K'])
    r_m_mod = (params['r_m'] * (1 + params['beta_h'] * sigma_n) +
               params['r_m'] * (params['beta_n'] * (C / params['K'])) +
               params['r_m'] * (params['beta_h_hat'] * sigma_n * (C / params['K'])))
    reaction_M = r_m_mod * M * (1 - M / params['K_m'])
    reaction_n = params['h1'] * params['v'] * (params['n0'] - n) - params['h2'] * C * n - params['h3'] * M * n

    dCdt = div_flux_C + reaction_C
    dMdt = div_flux_M + reaction_M
    dndt = div_flux_n + reaction_n

    return np.concatenate([dCdt, dMdt, dndt])

def run_simulation(params_input):
    """
    Run the PDE simulation using the provided parameters.
    params_input is a dictionary with keys for each parameter (except theta, n0, and h1).
    Returns spatial coordinate x and final profiles of C, M, and n.
    """
    # Spatial and temporal discretization
    L = 150.0    # domain length (mm)
    xh = 0.8
    x = np.arange(0, L + xh, xh)
    Nx = len(x)
    Tf = 720     # final time (days)

    # Initial conditions (n0 is now fixed to 1.0)
    C0 = 40 / (1 + np.exp(2*(x - 0.5)))
    M0 = 5 * np.ones_like(x)
    n0_arr = 1.0 * np.ones_like(x)
    U0 = np.concatenate([C0, M0, n0_arr])

    # Build parameters dictionary for simulation; theta, n0, and h1 are fixed.
    sim_params = {
        'D_C': params_input['D_C'],
        'D_m': params_input['D_m'],
        'r_c': params_input['r_c'],
        'r_m': params_input['r_m'],
        'D_n': params_input['D_n'],
        'alpha_h': params_input['alpha_h'],
        'alpha_n': params_input['alpha_n'],
        'alpha_h_hat': params_input['alpha_h_hat'],
        'beta_h': params_input['beta_h'],
        'beta_n': params_input['beta_n'],
        'beta_h_hat': params_input['beta_h_hat'],
        'gamma_h': params_input['gamma_h'],
        'gamma_n': params_input['gamma_n'],
        'gamma_h_hat': params_input['gamma_h_hat'],
        'delta_h': params_input['delta_h'],
        'delta_n': params_input['delta_n'],
        'delta_h_hat': params_input['delta_h_hat'],
        # Fixed parameters (removed from slicers):
        'theta': 20,         # fixed value
        'n0': 1.0,           # fixed value
        'h1': 0.337,         # fixed value
        'h2': params_input['h2'],
        'h3': params_input['h3'],
        # Other fixed parameters
        'K': 100,
        'K_m': 25,
        'n_cr': 0.25,
        'v': 1.0
    }

    sol = solve_ivp(lambda t, U: pde_rhs(t, U, sim_params, xh, Nx),
                    t_span=(0, Tf), y0=U0, t_eval=[Tf], method='BDF')
    if not sol.success:
        raise RuntimeError("Solver failed: " + sol.message)

    U_final = sol.y[:, -1].reshape(3, Nx)
    C_final, M_final, n_final = U_final[0, :], U_final[1, :], U_final[2, :]
    return x, C_final, M_final, n_final

###############################################################################
# Parameter slider definitions (each parameter as defined by its range, step, etc.)
# (Removed: theta, n0, h1)
###############################################################################
param_defs = [
    {"name": "D_C",        "min": 0.001, "max": 1.0,  "step": 0.005, "default": 0.2,   "scale": 1000},
    {"name": "D_m",        "min": 0.001, "max": 1.0,  "step": 0.005, "default": 0.3,   "scale": 1000},
    {"name": "r_c",        "min": 0.001, "max": 0.6,  "step": 0.005, "default": 0.02,  "scale": 1000},
    {"name": "r_m",        "min": 0.01,  "max": 1.0,  "step": 0.05,  "default": 0.4,   "scale": 100},
    {"name": "D_n",        "min": 50,    "max": 300,  "step": 10,    "default": 151,   "scale": 1},
    {"name": "alpha_h",    "min": 0.0,   "max": 2.0,  "step": 0.01,  "default": 0.5,   "scale": 100},
    {"name": "alpha_n",    "min": 0.0,   "max": 2.0,  "step": 0.01,  "default": 0.5,   "scale": 100},
    {"name": "alpha_h_hat","min": 0.0,   "max": 2.0,  "step": 0.01,  "default": 0.5,   "scale": 100},
    {"name": "beta_h",     "min": 0.0,   "max": 2.0,  "step": 0.01,  "default": 0.5,   "scale": 100},
    {"name": "beta_n",     "min": 0.0,   "max": 2.0,  "step": 0.01,  "default": 0.5,   "scale": 100},
    {"name": "beta_h_hat", "min": 0.0,   "max": 2.0,  "step": 0.01,  "default": 0.5,   "scale": 100},
    {"name": "gamma_h",    "min": 0.0,   "max": 2.0,  "step": 0.01,  "default": 0.5,   "scale": 100},
    {"name": "gamma_n",    "min": 0.0,   "max": 2.0,  "step": 0.01,  "default": 0.5,   "scale": 100},
    {"name": "gamma_h_hat","min": 0.0,   "max": 2.0,  "step": 0.01,  "default": 0.5,   "scale": 100},
    {"name": "delta_h",    "min": 0.0,   "max": 2.0,  "step": 0.01,  "default": 0.5,   "scale": 100},
    {"name": "delta_n",    "min": 0.0,   "max": 2.0,  "step": 0.01,  "default": 0.5,   "scale": 100},
    {"name": "delta_h_hat","min": 0.0,   "max": 2.0,  "step": 0.01,  "default": 0.5,   "scale": 100},
    {"name": "h2",         "min": 0.01,  "max": 0.2,  "step": 0.005, "default": 0.0598,"scale": 1000},
    {"name": "h3",         "min": 0.01,  "max": 0.1,  "step": 0.005, "default": 0.025, "scale": 1000}
]

###############################################################################
# Custom widget to encapsulate a slider with its label
###############################################################################
class ParameterSlider(QWidget):
    def __init__(self, name, min_val, max_val, step, default, scale, parent=None):
        super().__init__(parent)
        self.name = name
        self.scale = scale
        self.layout = QHBoxLayout(self)
        self.label = QLabel(f"{name} = {default:.3f}")
        self.slider = QSlider(Qt.Horizontal)
        # Set slider range in integer units
        self.slider.setMinimum(int(min_val * scale))
        self.slider.setMaximum(int(max_val * scale))
        self.slider.setSingleStep(int(step * scale))
        self.slider.setValue(int(default * scale))
        self.slider.valueChanged.connect(self.update_label)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.slider)

    def update_label(self, value):
        # Convert back to float value using scale
        float_val = value / self.scale
        self.label.setText(f"{self.name} = {float_val:.3f}")

    def get_value(self):
        return self.slider.value() / self.scale

###############################################################################
# Main application window
###############################################################################
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Glioma-Macrophage-Oxygen Simulation")

        # Main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        # Left panel: scrollable area for sliders
        scroll_widget = QWidget()
        self.sliders_layout = QVBoxLayout(scroll_widget)
        self.sliders = {}  # dictionary to store ParameterSlider objects

        # Create a slider for each parameter definition
        for pdef in param_defs:
            ps = ParameterSlider(
                pdef["name"],
                pdef["min"],
                pdef["max"],
                pdef["step"],
                pdef["default"],
                pdef["scale"]
            )
            self.sliders[pdef["name"]] = ps
            self.sliders_layout.addWidget(ps)

        self.sliders_layout.addStretch()

        from PyQt5.QtWidgets import QScrollArea
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_widget)

        # Right panel: matplotlib canvas
        self.fig = Figure(figsize=(6, 8))
        self.canvas = FigureCanvas(self.fig)

        # Button to run simulation
        self.run_button = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self.update_plot)

        # Arrange left panel: sliders and button
        left_layout = QVBoxLayout()
        left_layout.addWidget(scroll_area)
        left_layout.addWidget(self.run_button)

        # Add panels to the main layout
        main_layout.addLayout(left_layout, 1)
        main_layout.addWidget(self.canvas, 3)

        # Initial plot
        self.update_plot()

    def update_plot(self):
        # Gather parameters from sliders into a dictionary
        params = {name: slider.get_value() for name, slider in self.sliders.items()}
        try:
            x, C_final, M_final, n_final = run_simulation(params)
        except Exception as e:
            print("Simulation failed:", e)
            return

        # Clear figure and plot results
        self.fig.clear()
        ax1 = self.fig.add_subplot(3, 1, 1)
        ax1.plot(x, C_final, 'r-')
        ax1.set_ylabel("C (Glioma Cells)")
        ax1.set_title("Final Profile at Tf = 720 days")

        ax2 = self.fig.add_subplot(3, 1, 2)
        ax2.plot(x, M_final, 'b-')
        ax2.set_ylabel("M (Macrophages)")

        ax3 = self.fig.add_subplot(3, 1, 3)
        ax3.plot(x, n_final, 'g-')
        ax3.set_xlabel("x (mm)")
        ax3.set_ylabel("n (Oxygen)")

        self.fig.tight_layout()
        self.canvas.draw()

###############################################################################
# Main entry point
###############################################################################
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
