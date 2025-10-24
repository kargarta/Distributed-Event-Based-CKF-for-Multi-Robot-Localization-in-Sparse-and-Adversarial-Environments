# Event-Triggered CKF for Multi-Robot Localization in Sparse/Dense Enviroments
This repository implements an **Event-Triggered Cubature Kalman Filter (CKF)** for distributed multi-robot localization under **sparse sensing** and **dense sensing**. The framework integrates **Control Barrier Functions (CBFs)** for collision avoidance, **Stanley control** for leader–follower trajectory tracking.

---

## 🚀 Features

- **Event-triggered communication:** Adaptive thresholds to reduce redundant updates while maintaining estimation accuracy.  
- **Cubature Kalman Filter (CKF):** Nonlinear state estimation for robots with noisy range and bearing sensors.  
- **Multi-robot topology:** Dynamic Laplacian-based connectivity and neighbor detection.  
- **Control Barrier Functions (CBFs):** Safety-critical control for inter-robot collision avoidance.  
- **Leader–Follower Formation:** Stanley controller for smooth trajectory tracking.  
- **Simulation in Robotarium:** Compatible with `rps.robotarium` environment for realistic multi-robot execution.

---


---

## ⚙️ Dependencies

Make sure you have Python ≥ 3.8 and the following packages:

```bash
pip install numpy scipy matplotlib pandas
pip install robotarium


