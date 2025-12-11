# Data directory

This directory contains SPICE-generated results for all real testcases used for evaluation in ML-INSIGHT.

---

## 📌 Testcases overview

The folder `data/testcases/` contains SPICE-evaluated results for ten real designs from the ASAP7 technology node.  

To evaluate ML-INSIGHT, we synthesize 10 designs from open-cores, and OpenROAD GitHub repository in ASAP7 technology which vary in their number of instances (2,000 to over 200,000). 
Next, we perform floorplanning and placement using OpenROAD. We extract the leakage power, and the power grid and domain capacitance of each of these designs and treat each of them as individual gated domains. 
We determine the total number of switches per gated domain (or design) based on voltage drop requirements (5\% of VDD). 


Each file corresponds to one design and contains SPICE-ground-truth values used to train and validate ML-INSIGHT.

---

## Testcase metrics (ASAP7 Node)

| **Design** | **NT** | **# Instances** | **Cload (pF)** | **Ileak (µA)** |
|-----------|--------|------------------|----------------|----------------|
| Ariane | 1000 | 93,459  | 28.65 | 26.10 |
| AES256 | 930  | 233,579 | 22    | 23.71 |
| Mempool  | 710  | 130,460 | 19.49 | 17.40 |
| Ibex    | 500  | 54,785  | 2.78  | 3.04  |
| JPEG          | 440  | 53,317  | 6.13  | 6.57  |
| AES128  | 410  | 15,807  | 2.05  | 1.89  |
| Ethmac | 380  | 162,614 | 9.15  | 11.74 |
| Raven_SHA | 320  | 29,336  | 3.69  | 3.71  |
| Mock-array | 220 | 58,666 | 10.20 | 0.47 |
| Amber  | 200  | 2,747   | 1.00  | 0.68  |

---

## Files in `testcases/`

Each file corresponds to a design and contains SPICE-measured peak inrush current and wakeup latency values for different PSN patterns, corresponding to the total switches and total stages for each design.
These files are used in evaluating the ML models and SA framework used in ML-INSIGHT.


