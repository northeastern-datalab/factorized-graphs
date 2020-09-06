## Factorized Graph Representations for Semi-supervised Learning from Sparse Labels

[![SIGMOD](https://img.shields.io/badge/SIGMOD-2020-blue.svg)](https://doi.org/10.1145/3318464.3380577)
[![Paper](http://img.shields.io/badge/paper-arxiv.2003.02829-blue.svg)](https://arxiv.org/abs/2003.02829)
[![Python 3.6](https://img.shields.io/badge/python-3.6-orange.svg)](https://www.python.org/downloads/release/python-360/)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](https://opensource.org/licenses/Apache-2.0)



This library provides various Python modules and scripts to perform semi-supervised learning with heterophily (SSLH). 
It includes methods to perform label propagation with linearized belief propagation and to estimate class-to-class compatibilities from very sparsely labeled graphs. 
Also included is code and experimental traces to reproduce the experiments from our [SIGMOD 2020](http://sigmod2020.org/) paper:
[Factorized Graph Representations for Semi-supervised Learning from Sparse Labels](https://doi.org/10.1145/3318464.3380577)

**Overview of SSLH:** 
Given a partially labeled graph and a class-to-class compatibility matrix, linearized belief propagation (LinBP) performs a generalized form of label propagation to label the remaining nodes.
Distant compatibility estimation (DCE) performs the same function but *does not require the compatibility matrix as input*. 
For quick understanding of the approach, please also see the video presented at SIGMOD 2020:

[![Watch the video](https://i.imgur.com/5DhXopX.png)](https://www.youtube.com/watch?v=t6ajKsZRt0o&list=PL_72ERGKF6DTTD6T5oR4WQPuCyHZd7x_N)


## Dependencies
Dependencies can be installed using `requirements.txt`.


## Project structure

  - `experiments_sigmod20/` folder containing scripts and notebooks for recreating figures from the paper
    - `datacache/` folder containing traces from experiments saved as CSV
    - `figs/` folder in which code places figures from experiments  
    - `realData/` place real data sets into this folder before running experiments
    - `...` various modules that perform varous experiments 
    - `Figures_realdata_sigmod20.ipynb` Notebook that plots all figures for experiments on 8 real data sets  
    - `Figures_syntheticdata_sigmod20.ipynb` **Start here**: Notebook that plots all other figures in the paper
  - `sslh/` folder containing modules with main functions 
    - `estimation.py` module containing main functions for parameter estimation
    - `fileInteraction.py` module containing functions for loading and saving experimental results
    - `graphGenerator.py` module containing synthetic graph generator with planted graph properties
    - `inference.py` module containing main propagation methods for linearized belief propagation
    - `utils.py` module containing various helper functions
    - `visualize.py` helper function to plot figures
  - `test_sslh/` folder with unit tests for modules and functions in `sslh/`


## Real data sets
A copy of the 8 real datasets we used in our experiments is available in the orm of 16 CSV files totaling 1.2GB on [Google Drive](https://drive.google.com/drive/folders/1fqTgfW8f-PUwnAj432YgsFVjgbUdOHuu?usp=sharing). 
To run the experiments, place them into the folder `experiments_sigmod20/realData/`, then run the respective methods in `experiments_sigmod20/`.


## Usage
For examples on the usage of the various methods, please see the `test_sslh` directory in the source tree.


## License
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)


## Citation
If you use this code in your work, please cite: 
```bibtex
@inproceedings{DBLP:conf/sigmod/PLG20,
  author    = {Krishna Kumar P. and Paul Langton and Wolfgang Gatterbauer},
  title     = {Factorized Graph Representations for Semi-Supervised Learning from Sparse Data},
  booktitle = {International Conference on Management of Data (SIGMOD)},
  pages     = {1383--1398},
  publisher = {{ACM}},
  year      = {2020},
  url       = {https://doi.org/10.1145/3318464.3380577},
}
```

## Contributors
- [Wolfgang Gatterbauer](http://gatterbauer.name)
- Krishna Kumar
- [Paul Langton](https://github.com/paulangton)

For any clarification, comments, or suggestions on the main methods in `sslh/` please create an issue or contact [Wolfgang](http://gatterbauer.name).
For any questions on the scripts in `experiments_sigmod/` and reproducability of the experiments, please contact [Paul](https://github.com/paulangton).