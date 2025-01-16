[![Doctests](https://github.com/notna07/disjoint-synthetic-data-generation/actions/workflows/doctests.yml/badge.svg)](https://github.com/notna07/disjoint-synthetic-data-generation/actions/workflows/doctests.yml)

# Disjoint Generative Models 

Disjoint Generative Models (DGMs) is a framework for generating synthetic data by distributing the generation of different attributes to different generative models. This has the benefit of being able to choose the ``correct tool for the correct job'' and infers increased privacy by not having a single model that has access to all the data.

The library provides a simple API for generating synthetic data using a variety of generative models and joining strategies. The library has access to a variety of generative model backends namely [SynthCity](https://github.com/vanderschaarlab/synthcity), [DataSynthesizer](https://github.com/DataResponsibly/DataSynthesizer) and [Synthpop](https://www.synthpop.org.uk/get-started.html), but additional backends can be added in the adapters module. Similarly several methods for joining are available for combining the generated data, and more can be added in the joining strategies module.

## Installation

To install the library, run the following command:

```bash
pip install 
```
One of the generative model backends "synthpop" requires a working R installation on the system. Access is handled through ```subprocess``` to run an ```Rscript``` command, so make sure that the Rscript command works in the terminal.

## Useage Examples (Experiments in the Paper)
 
Below is codebooks that can be used to replicate the results shown in the paper.
| Link | Description |
| --- | --- |
| [Tutorial](00_tutorial.ipynb) | A simple tutorial on how to use the library |
| [Case Study 2](03_specified_splits.ipynb) | Mixed Models has Significant Impact on Privacy |


# TODO
- [ ] reconsider if we will be using rpy2 or not
- [ ] Consider default options to make user interface simpler for common use cases
- [ ] Add more generative model backends
- [ ] Add more joining strategies
- [ ] Add more experiments
