# **ML-INSIGHT : Machine Learning–Driven Power Switch Network Optimization**  
[[ML-INSIGHT V1 (ISLPED 2024)]](https://dl.acm.org/doi/10.1145/3665314.3670807)

ML-INSIGHT is a machine-learning–based framework that accelerates **Power Switch Network (PSN)** optimization by predicting **inrush current** and **wake-up latency** with over **50× speedup** compared to SPICE. The work in this repository is an extension of the ML-INSIGHT paper published in ISLPED 2024.

## **Abstract**

Power gating is widely used to reduce leakage power in modern SoCs, but powering up a gated domain causes inrush current surge, which must be kept under strict limits. To avoid high inrush current, engineers stagger the power switches into multiple stages, but having an increased number of stages will lead to higher wakeup latency, which is also not desirable. Hence it is highly important to find the optimal power switch netowrk (PSN) pattern which has minimal inrush and wakeup latency. Traditional SPICE simulations accurately estimate these metrics but are computationally expensive, making full PSN design-space exploration infeasible. ML-INSIGHT introduces two machine-learning models—one predicting peak inrush current, the other predicting wakeup latency—that replace SPICE during PSN exploration. The inrush model uses a cascading prediction algorithm that generalizes across arbitrary numbers of stages; the latency model uses statistical + positional features to remain stage-independent. Integrated with a simulated annealing optimizer, ML-INSIGHT rapidly identifies PSN patterns that not only meet design constraints but also minimizes peak inrush current and wakeup latency observed during activation.

Experiments using ten real designs in ASAP7 show:  
- <10% mean prediction error for both metrics
- \>50× speedup over SPICE  
- Generalization across designs, domain sizes, and PSN patterns  

---

## **Table of Contents**

- [Repo Structure](#repo-structure)  
- [Getting Started](#getting-started)  
- [Prerequisites](#prerequisites)  

---

## **Repository Structure**

- [data](./data/)
  - [testcases](./data/testcases/)

- [models](./models/)

- [src](./src/)
  - [analytical_model](./src/analytical_model/)
  - [ML-INSIGHT](./src/ML-INSIGHT/)
  - [ML-INSIGHT_v1](./src/ML-INSIGHT_v1/)

- [README.md](./README.md)

## Getting Started

Follow the steps below to install and run ML-INSIGHT

### 1. Clone the Repository
```
git clone https://github.com/ASU-VDA-Lab/PowerSwitch-ML-INSIGHT.git
cd PowerSwitch-ML-INSIGHT
```

### 2. Install Dependencies
Create a Python virtual environment and install dependencies
```
python3 -m venv mlinsight-env
source mlinsight-env/bin/activate
pip install -r requirements.txt
```
---
