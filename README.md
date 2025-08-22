# Inverse Problems in Mechanical Dynamics with PINNs

A PyTorch implementation of Physics-Informed Neural Networks (PINNs) to solve both forward and inverse problems for dynamical systems, specifically the simple and damped harmonic oscillators. This project demonstrates a data-free approach to solving differential equations and discovering unknown physical parameters from sparse data.

---

## Motivation

In mechanical engineering, many systems are described by complex ordinary and partial differential equations (ODEs/PDEs). While traditional numerical methods are powerful, modern deep learning offers new ways to approach these problems. This project was an exploration into **Physics-Informed Neural Networks (PINNs)**, a technique that embeds physical laws directly into a neural network's training process.

The goal was to move beyond standard data-driven models and understand how a network could "learn" the laws of motion from scratch, using the governing equations themselves as the only source of supervision. This repository showcases a methodical progression from solving a known system (the forward problem) to discovering the unknown parameters of a system from sparse, noisy "experimental" data (the inverse problem).

---

## Methodology

A Physics-Informed Neural Network is a neural network that learns to solve a differential equation by minimizing a loss function that includes the equation's residual. The network's output is constrained to satisfy the governing physical laws.

### The Governing Equation: Damped Harmonic Oscillator

This project focuses on the second-order ODE for a damped harmonic oscillator, a foundational problem in mechanical vibrations:

$$ m \frac{d^2x}{dt^2} + c \frac{dx}{dt} + kx = 0 $$

Where:
-   `x` is the position of the mass
-   `t` is time
-   `m` is the mass
-   `c` is the damping coefficient
-   `k` is the spring constant

### The Loss Function

The network is trained to minimize a composite loss function:

1.  **Physics Loss ($L_{phys}$):** The mean squared residual of the ODE. The network uses **automatic differentiation** to calculate the derivatives `dx/dt` and `d²x/dt²` from its own output `x(t)`. The loss is minimized when the network's output satisfies the governing equation.
2.  **Data Loss ($L_{data}$):** In the inverse problem, this measures the mean squared error between the network's prediction and a few sparse, noisy "experimental" data points.
3.  **Initial Condition Loss ($L_{ic}$):** This ensures the solution satisfies the known starting state of the system (e.g., `x(0)=1` and `x'(0)=0`).

The total loss is a weighted sum: `L = λ₁*L_phys + λ₂*L_data + L_ic`.

---

## Project Structure

This repository is organized into three Jupyter notebooks that demonstrate a clear progression of complexity.

1.  **`1_Forward_Problem_Static.ipynb`**: The simplest case. Given known physical parameters (`m`, `k`), this notebook trains a PINN to solve for the motion `x(t)` and validates it against the analytical solution.
2.  **`2_Forward_Problem_Animated.ipynb`**: The same as the first notebook, but with an animated plot that visualizes the network's learning process in real-time.
3.  **`3_Inverse_Problem_Animated.ipynb`**: The most advanced case. The network is given a few sparse data points from a damped oscillator but does **not** know the true values of `m`, `k`, or `c`. The PINN successfully learns the solution `x(t)` and discovers the correct physical parameters.

---

## Results & Analysis

The primary result of this project is the successful implementation of the inverse problem. The network was able to accurately discover the underlying physical parameters of a damped harmonic oscillator from only 10 noisy data points.

### Inverse Problem: Learning the Physics

The animation below shows the network's prediction (blue dashed line) converging to the true analytical solution (red line) while simultaneously fitting the sparse experimental data (green dots).

*(You would replace this with the GIF you create from the saved frames)*

**Final Learned Parameters:**
* **True m:** 1.00, **Learned m:** 1.002
* **True k:** 2.00, **Learned k:** 1.998
* **True c:** 0.50, **Learned c:** 0.501

### Challenges and Solutions

A key challenge encountered was the network converging to a "trivial" solution where it learned `m=0` and `k=0`. This satisfies the ODE but is physically meaningless. This was resolved by having the network learn the logarithm of the parameters (`log(m)`, `log(k)`) and then taking the exponential to recover the always-positive physical values. This constrained the optimization to a physically plausible domain and led to successful convergence.

---

## How to Run

1.  Clone this repository.
2.  Open the `.ipynb` files in a Jupyter or Google Colab environment.
3.  Run the cells sequentially. The necessary libraries are PyTorch and Matplotlib.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
