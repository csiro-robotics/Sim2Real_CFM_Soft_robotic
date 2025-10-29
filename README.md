# Soft Robotic Sim2Real via Conditional Flow Matching

This repository contains the implementation of our method for bridging the Sim2Real gap in soft robotics using Conditional Flow Matching (CFM). Our approach learns a mapping between simulation and real-world domains to accurately predict force and deformation in soft robotic systems.

## Abstract

Modelling soft robots remains a significant challenge due to high computational costs and frequent mismatches with real-world behaviour; a phenomenon known as the Sim2Real gap. This paper addresses the Sim2Real gap through Conditional Flow Matching, which learns a mapping between the simulation domain and the real-world experimental domain. A neural network learns a conditional probability path that transforms simulated states into real-world observations, conditioned on control inputs, thereby minimising simulation inaccuracies. The method is demonstrated through benchmark Sim2Sim and Sim2Real tensile tests, and additionally demonstrated in the domain of soft gripping using Fin Ray grippers, for which we introduce a novel encoder architecture that learns a representation of the contact state, enabling the model to generalize to previously unseen interactions. The model provides highly accurate prediction of force and deformation, successfully capturing complex elastic behaviours including hysteresis and force fluctuations. Experimental results validate that Conditional Flow Matching can bridge the Sim2Real gap for various soft robot morphologies, without requiring large datasets, and with strong generalisation capabilities.

## Features

- **Conditional Flow Matching**: Novel application of CFM to bridge the Sim2Real gap in soft robotics
- **Multi-Domain Support**: Works with both tensile test benchmarks and Fin Ray gripper systems
- **Encoder Architecture**: Custom encoder for learning contact state representations
- **Hysteresis Modeling**: Captures complex elastic behaviors including hysteresis and force fluctuations
- **Strong Generalization**: Effective performance with limited datasets and on unseen interactions

## Repository Structure

```
├── flow_model/                          # Core flow matching models
│   ├── conditional_flow.py              # Conditional vector field implementation
│   ├── conditional_flow_match.py        # Flow matching algorithms
│   └── data_model.py                    # Force predictor neural network
│
├── utils/                               # Utility functions
│   ├── data_funcs.py                    # General data processing utilities
│   └── data_funcs_tensile_sim2real_gen.py  # Tensile test data processing
│
├── tensile-sim2real-src/                # Tensile test experiments
│   ├── data_gen_model_cfm_extra_condi.py   # CFM training for tensile tests
│   ├── data_tensile_exp/                # Experimental tensile data
│   ├── data_tensile_warp/               # Simulated tensile data
│   ├── data_tensile_exp_val/            # Experimental validation data
│   └── data_tensile_warp_val/           # Simulated validation data
│
├── dataset_model_sim2real/              # Trained models (*.pth files)
├── dataset_model/                       # Additional model checkpoints
│
└── README.md                            # This file
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- PyTorch 1.12+

### Setup

1. Clone the repository:
```bash
git clone https://github.com/csiro-robotics/Sim2Real_CFM_Soft_robotic.git
cd Sim2Real_CFM_Soft_robotic
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the CFM Model

Run the training script for tensile test Sim2Real:

```bash
cd tensile-sim2real-src
python data_gen_model_cfm_extra_condi.py
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{yourpaper2024,
  title={Soft Robotic Sim2Real via Conditional Flow Matching},
  author={Your Names},
  journal={Journal Name},
  year={2024}
}
```

## License

Apache License

## Contact

For questions and feedback, please contact:
- [Ge Shi] - [ge.shi@csiro.au]

## Acknowledgments

This research was conducted at CSIRO's Robotics and Autonomous Systems Group.