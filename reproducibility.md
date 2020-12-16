# ACM SIGMOD 2021 Reproducibility

This page contains a detailed description to reproduce the experimental results reported in SIGMOD 2020 paper **Factorized Graph Representations for Semi-Supervised Learning from Sparse Data** as submitted to the [ACM SIGMOD 2021 Reproducibility](https://reproducibility.sigmod.org/). 

The published paper is available in the [ACM Digital Library](https://dl.acm.org/doi/pdf/10.1145/3318464.3380577). The full version is available on [arXiV.2003.02829](https://arxiv.org/pdf/2003.02829.pdf). 


**Programming Language:** Python3   
**Dependency Packages**:
```
jupyter>=1.0.0
matplotlib>=1.4.2
numpy>=1.9.1
networkx>=1.11
pandas>=0.19.0
pyamg>=2.2.1
pytest>=2.8.0
scipy>=0.15.1
scikit-learn>=0.18
sklearn
seaborn>=0.8.0
```
Requirements referenced in Github [requirements.txt](https://github.com/northeastern-datalab/factorized-graphs/blob/master/requirements.txt)  
To install dependencies, simply run:  `pip install -r requirements.txt`



### B) Datasets info
**Data generators**: Github repository has our [synthetic data generators](https://github.com/northeastern-datalab/factorized-graphs/blob/master/sslh/graphGenerator.py). It is not required to separately generate data, since the scripts to run our experiments auto-generate the requisite data.   
**Real dataset Repository**: 8 real datasets we used in our experiments are available in the form of 16 CSV files totaling `1.2GB` on [Google Drive](https://drive.google.com/drive/folders/1fqTgfW8f-PUwnAj432YgsFVjgbUdOHuu). 
Download the real datasets and copy it into the following directory:`factorized-graphs/experiments_sigmod20/realData/`



### C) Hardware Info   

Experiments were primarily run on a MacBook Pro with the below configuration. Some of the real-world large datasets were run on clusters, detailed below. 

Hardware for all Timing Experiments (including Fig.3,  Fig.5, Fig.6 in the [paper](https://dl.acm.org/doi/pdf/10.1145/3318464.3380577))   
  C1) *Processor*: 2.5 GHz Intel Core i5   
  C2) *Caches* (number of levels, and size of each level)   
  C3) *Memory*: 16 GB  
  C4) *Secondary Storage*:   1 TB SSD    
  C5) *Network* (if applicable: type and bandwidth)   


Hardware for Accuracy Estimation Experiments on real-world datasets  (Fig.7)
(For detailed specs, please refer [The Discovery Cluster](https://rc-docs.northeastern.edu/en/latest/welcome/welcome.html) at [MGHPCC](https://www.mghpcc.org/))   
  C6) *Processor* (architecture, type) 2.4 GHz Intel E5-2680 v4 CPUs   
  C7) *Caches* (number of levels, and size of each level)    
  C8) *Memory* (size and speed): 256 GB/node   
  C9) *Secondary Storage* (type: SSD/HDD/other, size, performance: random read/sequential read/random write/sequential write)   GPFS, disk type unavailable, assume SSD since it’s a world-class HPC facility.    
  C10) *Network* (if applicable: type and bandwidth) InfiniBand (IB) interconnect running at 100 Gbps



### D) Experimentation Info
All experiments can easily be run using two Jupyter notebooks and recreate all the figures in our paper.
These two notebooks, one for synthetic data and the other for real datasets, are meant to serve as a quick-access portal into the rest of the code using figures in the paper as point of reference. 

* Link to [Jupyter notebook A](https://github.com/northeastern-datalab/factorized-graphs/blob/master/experiments_sigmod20/Figures_syntheticdata_sigmod20.ipynb) for theoretical verification and synthetic data experiments.
* Link to [Jupyter notebook B](https://github.com/northeastern-datalab/factorized-graphs/blob/master/experiments_sigmod20/Figures_realdata_sigmod20.ipynb) for experiments on real-world datasets.
 
Run the Jupyter notebook and jump to the necessary functions to learn how a certain figure was generated in the paper.

**D1) Scripts and how-tos to generate all necessary data or locate datasets**   
The code allows two levels of granularity to reproduce all results:
  1. Reproduce figures using data cache: We have cached all the intermediate results required to produce the figures in our paper in `factorized-graphs/experiments_sigmod20/datacache` subfolders. You can simply run the cells in Jupyter notebook to generate figures. Those are the defaults for the provided notebooks, and we recommend this mode during the first pass.
  2. In order to run all the experiments from scratch, please use the option “create_data =  True” in  the provided Jupyter notebooks. On a 2018 Macbook pro, this would  take approximately ###$$$ hours for the synthetic, and ###$$$ hours for the real data experiments. We suggest to thus reproduce the graphs with less accuracy instead (fewer data samples and thus more wiggly) by using changing the parameters ###$$$ in the notebook

#### Example: 
* To generate Fig 5(a) using cached intermediate data, run   
  `Fig_Backtracking_Advantage.run(choice=31, variant=0, create_data=False, show_plot=True, create_pdf=True)`   
* To ignore the cached data and recreate figures from scratch  
  `Fig_Backtracking_Advantage.run(choice=31, variant=0, create_data=True, show_plot=True, create_pdf=True)`  
 
*Note:*  choice and variant parameters are to select between different configurations of graphs.   


**D2) Scripts and how-tos to prepare the software for system**   
Since our code base is python based, it is very simple to prepare the system; just install the library dependencies using `pip install -r requirements.txt`


**D3) Scripts and how-tos for all experiments executed for the paper**   
Run the cells in Jupyter notebook. 
Expected time to run a cell in notebook: ###$$$  

*Please note:* Many of the accuracy experiments over real datasets (Figure.7) were run at [MGHPCC](https://www.mghpcc.org/) on high-performance compute infrastructure. Although the plots are reproducible instantaneously from cached data provided in the repository, if you wish to recreate the experimental data points from scratch (i.e. using the `create_data=True` flag), it is highly recommended to run the real data experiments with a large number of CPU cores and plenty of memory. 
They are feasible to run on a home computer, but it will likely take multiple days to produce plots with comparable variance to those presented in the paper.
