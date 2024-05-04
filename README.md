This repository contains the source code for the experiments done in the master thesis: 
### Algorithm Engineering and Empirical Evaluation for Fast Rerouting with Multiple Failures
written by Philipp Zabka

## Overview

This repository contains the following scripts:
1. zabka2024_experiments.py: Experiments performed for this thesis
2. TwoResilientAlg.py: The TwoResilient algorithm
3. FeigenbaumAlg.py: The Feigenbaum algorithm 
4. graph_utils.py: Helper functions for reading, writing and creating graphs
5. routing.py: Routing algorithms, simulation and statistic framework (also contains the SquareOne algorithm)
6. arborescences.py: Arborescence decomposition and helper algorithms

## Run experiments

The following section describes how to run each the experiments on each network topology

### Zoo

For this experiment, the networks from [Internet Topology Zoo](http://www.topology-zoo.org) have to be downloaded and copied into the directory `benchmark_graphs/zoo`.
Results will be generated in the `results_zoo` directory.

Example call:
```shell
python zabka2024_experiments.py zoo 45 10 100 20 20 RANDOM False
```
Refer to `zabka2024_experiments.py` for a description of the input parameters.

### Rocketfuel

For this experiment, the networks from [Rocketfuel](https://research.cs.washington.edu/networking/rocketfuel/) have to be downloaded and copied into the directory `benchmark_graphs/rocket_fuel`.
Results will be generated in the `results_as` directory.

Example call:

```shell
python zabka2024_experiments.py as 45 10 100 20 20 RANDOM False
```

### Erdős–Rényi

Networks will be automatically generated in directory `benchmark_graphs/erdos-renyi`.
Results will be generated in the `results_erdos_renyi` directory.

Example call:

```shell
python zabka2024_experiments.py erdos-renyi 45 10 100 20 20 RANDOM False
```

### Regular

Networks will be automatically generated in directory `benchmark_graphs/regular`.
Results will be generated in the `results_regular` directory.

Example call:

```shell
python zabka2024_experiments.py regular 45 10 100 20 20 RANDOM False
```

### Ring

Networks will be automatically generated in directory `benchmark_graphs/ring`.
Results will be generated in the `results_ring` directory.

Example call:

```shell
python zabka2024_experiments.py ring 45 10 100 20 20 RANDOM False
```

## Addendum

Some of the source code provided in this repository builds upon source code originating from [Fast-Failover](https://gitlab.cs.univie.ac.at/ct-papers/fast-failover/-/tree/master). 

December 2020: Paula-Elena Gheorghe added an implementation of the algorithm described in https://cpsc.yale.edu/sites/default/files/files/tr1454.pdf, see FeigenbaumAlgo.py
