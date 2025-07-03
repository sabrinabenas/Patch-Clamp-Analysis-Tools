# Patch Clamp Analysis Tools

This repository includes three Python scripts designed for the analysis of patch clamp electrophysiological recordings in ABF (Axon Binary Format). Each tool focuses on a specific type of analysis commonly used in neurophysiology research.

---

## 1. Patch Clamp Analysis Tool for Input Resistance and Capacitance Measurement

This script analyzes ABF files from patch clamp recordings to calculate key membrane properties:

- **Input Resistance**
- **Membrane Capacitance**
- **Spike Amplitude**

These metrics provide essential insights into the passive properties of neurons or other electrically excitable cells.

---

## 2. Spike Analysis Tool for Patch Clamp Recordings

This script focuses on detecting and characterizing action potentials (spikes) in ABF files. It extracts a wide range of features, including:

- **Spike Amplitude**
- **Afterhyperpolarization (AHP) Depth**
- **Voltage Threshold**
- **Spike Width and Timing**
- **Spike Count and Frequency**

It combines the **eFEL** (Electrophys Feature Extraction Library) with custom algorithms for precise threshold detection and robust feature extraction.

---

## 3. Spontaneous Event Analysis Tool for Patch Clamp Recordings

This script analyzes ABF files to detect and characterize spontaneous synaptic events (such as sEPSCs or sIPSCs). It measures:

- **Event Amplitude and Frequency**
- **Event Kinetics** (Rise Time, Decay Tau, Width)
- **Area Under the Curve (AUC)**
- **Artifact Removal and Signal Cleaning**

The script uses adaptive filtering and prominence-based peak detection to accurately identify events in noisy electrophysiological recordings.

---

## Requirements

- Python 3.x
- pyabf (for ABF file handling)
- eFEL (for spike feature extraction)
- scipy, numpy, matplotlib

Install dependencies with:

```bash
pip install pyabf efel scipy numpy matplotlib
```

---

## Usage

Example:

```bash
python spike_analysis.py 
```

Output includes numerical results and optionally visualizations of the traces and detected features.

