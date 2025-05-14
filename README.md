<!-- [![Doctests](https://github.com/notna07/disjoint-synthetic-data-generation/actions/workflows/doctests.yml/badge.svg)](https://github.com/notna07/disjoint-synthetic-data-generation/actions/workflows/doctests.yml) -->

# Disjoint Generative Models 

Disjoint Generative Models (DGMs) is a framework for generating synthetic data by distributing the generation of different attributes to different generative models. DGMs unlock mixed model generation, allowing the user to choose ``correct tool for the correct job'' and infers increased privacy by not having a single model that has access to all the data.

The library provides a simple API for generating synthetic data using a variety of generative models and joining strategies. The library has access to a variety of generative model backends namely [SynthCity](https://github.com/vanderschaarlab/synthcity), [DataSynthesizer](https://github.com/DataResponsibly/DataSynthesizer) and [Synthpop](https://www.synthpop.org.uk/get-started.html), but additional backends can be added in the adapters module. Similarly several methods for joining are available for combining the generated data, and more can be added in the joining strategies module.

## Installation

 To install the library, run the following command:

```bash
pip install 
```
One of the generative model backends "synthpop" requires a working R installation on the system. Access is handled through ```subprocess``` to run an ```Rscript``` command, so make sure that the Rscript command works in the terminal.

## Tutorial and Codebooks
 
Below is codebooks that can be used to replicate the results shown in the paper.
| Link | Description | Fig. refs. |
| --- | --- | --- |
| [Tutorial](00_tutorial.ipynb) | A simple tutorial on how to use the library | NA |
| [Codebook 1](01_same_model_partitions.ipynb) | Introductionary experiments, random joining, incresing number of partitions | Fig.2 |
| [Codebook 2](02_validated_joins.ipynb) | High-dimensional dataset example vith validation, correlated partitions study | Fig. 3, 4, 5 |
| [Codebook 3](03_specified_splits.ipynb) | Mixed-model generation and combinatorics | Fig. 6, Tab. 2, 3 |
| [Codebook 4](04_joining_validator.ipynb) | Study of the joining validator model, optimisation and calibration | Fig. 7, 8, 9 |

Additional examples for how to use the library can be seen in the documentation in the source code folder. 

## Requirements
The library requires Python 3.10 or higher and the following packages:
- numpy
- pandas
- scipy ~= 1.12
- scikit-learn ~= 1.5
- synthcity >= 0.2.11
- DataSynthesizer ~= 0.1.13
- pyod >= 2.0

Additonally, the synthpop generative model is accessed through R, and requires the following R packages:
- synthpop