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
python zabka2024_experiments.py zoo 45 100 100 20 20 RANDOM False
```
Short description of arguments:
1. zoo: Switch -> decides which experiments to run
2. 45: Seed -> Random seed, used for graph generation
3. 100: Repetitions -> Number of experiments
4. 100: Number of nodes -> Set number of nodes for graph generation
5. 20: Sample size -> Number of sources to route a packet to destination
6. 20: Number of failed link -> Set number of failed links for the experiment
7. False: Short -> If True, only small zoo graphs < 25 nodes are run 

Refer to `zabka2024_experiments.py` for more information on input arguments.

### Rocketfuel

For this experiment, the networks from [Rocketfuel](https://research.cs.washington.edu/networking/rocketfuel/) have to be downloaded and copied into the directory `benchmark_graphs/rocket_fuel`.
Results will be generated in the `results_as` directory.

Example call:

```shell
python zabka2024_experiments.py as 45 100 100 20 20 RANDOM False
```

### Erdős–Rényi

Networks will be automatically generated in directory `benchmark_graphs/erdos-renyi`.
Results will be generated in the `results_erdos_renyi` directory.

Example call:

```shell
python zabka2024_experiments.py erdos-renyi 45 100 100 20 20 RANDOM False
```

### Regular

Networks will be automatically generated in directory `benchmark_graphs/regular`.
Results will be generated in the `results_regular` directory.

Example call:

```shell
python zabka2024_experiments.py regular 45 100 100 20 20 RANDOM False
```

### Ring

Networks will be automatically generated in directory `benchmark_graphs/ring`.
Results will be generated in the `results_ring` directory.

Example call:

```shell
python zabka2024_experiments.py ring 45 100 100 20 20 RANDOM False
```

## Addendum

Some of the source code provided in this repository builds upon source code originating from [Fast-Failover](https://gitlab.cs.univie.ac.at/ct-papers/fast-failover/-/tree/master). 

December 2020: Paula-Elena Gheorghe added an implementation of the algorithm described in https://cpsc.yale.edu/sites/default/files/files/tr1454.pdf, see FeigenbaumAlgo.py

If you use this the code from TwoResilient, Feigenbaum or SquareOne, please cite the corresponding paper(s).

#### TwoResilient
```
@inproceedings{DBLP:conf/spaa/DaiF023,
  author       = {Wenkai Dai and
                  Klaus{-}Tycho Foerster and
                  Stefan Schmid},
  editor       = {Kunal Agrawal and
                  Julian Shun},
  title        = {A Tight Characterization of Fast Failover Routing: Resiliency to Two
                  Link Failures is Possible},
  booktitle    = {Proceedings of the 35th {ACM} Symposium on Parallelism in Algorithms
                  and Architectures, {SPAA} 2023, Orlando, FL, USA, June 17-19, 2023},
  pages        = {153--163},
  publisher    = {{ACM}},
  year         = {2023},
  url          = {https://doi.org/10.1145/3558481.3591080},
  doi          = {10.1145/3558481.3591080},
  timestamp    = {Thu, 15 Jun 2023 21:57:01 +0200},
  biburl       = {https://dblp.org/rec/conf/spaa/DaiF023.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

#### SquareOne
```
@inproceedings{DBLP:conf/infocom/FoersterP0T19,
  author       = {Klaus{-}Tycho Foerster and
                  Yvonne{-}Anne Pignolet and
                  Stefan Schmid and
                  Gilles Tr{\'{e}}dan},
  title        = {{CASA:} Congestion and Stretch Aware Static Fast Rerouting},
  booktitle    = {2019 {IEEE} Conference on Computer Communications, {INFOCOM} 2019,
                  Paris, France, April 29 - May 2, 2019},
  pages        = {469--477},
  publisher    = {{IEEE}},
  year         = {2019},
  url          = {https://doi.org/10.1109/INFOCOM.2019.8737438},
  doi          = {10.1109/INFOCOM.2019.8737438},
  timestamp    = {Wed, 17 Mar 2021 13:51:52 +0100},
  biburl       = {https://dblp.org/rec/conf/infocom/FoersterP0T19.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

#### Feigenbaum
```
@article{DBLP:journals/corr/abs-1207-3732,
  author       = {Joan Feigenbaum and
                  Brighten Godfrey and
                  Aurojit Panda and
                  Michael Schapira and
                  Scott Shenker and
                  Ankit Singla},
  title        = {On the Resilience of Routing Tables},
  journal      = {CoRR},
  volume       = {abs/1207.3732},
  year         = {2012},
  url          = {http://arxiv.org/abs/1207.3732},
  eprinttype    = {arXiv},
  eprint       = {1207.3732},
  timestamp    = {Mon, 13 Aug 2018 16:48:07 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1207-3732.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

