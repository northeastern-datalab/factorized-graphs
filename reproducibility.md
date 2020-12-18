# Reproducibility of SIGMOD 2020 Experiments

This page contains a detailed description to reproduce the experimental results reported 
in the SIGMOD 2020 paper #125 titled 
[*Factorized Graph Representations for Semi-Supervised Learning from Sparse Data*](https://dl.acm.org/doi/pdf/10.1145/3318464.3380577) 
as submitted to the [ACM SIGMOD 2021 Reproducibility Track](https://reproducibility.sigmod.org/). 



### Research Paper
The official paper is available in the 
[ACM Digital Library (https://dl.acm.org/doi/10.1145/3318464.3380577)](https://dl.acm.org/doi/10.1145/3318464.3380577). 
The full version is available on [arXiV.2003.02829](https://arxiv.org/abs/2003.02829). 
For citing our work, we suggest using the [DBLP bib file](https://dblp.org/rec/conf/sigmod/PLG20.html?view=bibtex).



### Programming Language and Dependencies
All code is implemented in Python3 with following dependencies 
(also listed in [`requirements.txt`](https://github.com/northeastern-datalab/factorized-graphs/blob/master/requirements.txt)):  
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
To install all dependencies, simply run:  `pip install -r requirements.txt`



### Datasets Used
**Synthetic data generator**: 
Our github repository includes our synthetic data generators 
[`/sslh/graphGenerator.py`](https://github.com/northeastern-datalab/factorized-graphs/blob/master/sslh/graphGenerator.py). 
It is not required to separately generate data, as the experimental scripts auto-generate the necessary  data.   

**Real dataset Repository**: We used 8 real datasets in our experiments that are available in the form of 
16 CSV files totaling `1.2GB` in a separate [Google Drive folder](https://drive.google.com/drive/folders/1fqTgfW8f-PUwnAj432YgsFVjgbUdOHuu). 
Please download those real datasets and copy them into the following directory before running the experiments on real data: 
`/experiments_sigmod20/realData/`



### Hardware Info   
Experiments were primarily run on a 2016 MacBook Pro 13-inch with the below configuration ("Hardware 1"). 
Some of the real-world large datasets were run on a cluster, detailed below ("Hardware 2").

**Hardware 1**: used for all Timing Experiments (including Figures 3, 5, an 6 in the [paper](https://dl.acm.org/doi/pdf/10.1145/3318464.3380577)) :  
- *Processor*: 2.5 GHz Intel Core i5   
- *Memory*: 16 GB  
- *Secondary Storage*:   1 TB SSD       


**Hardware 2**: used for Accuracy Estimation Experiments on real-world datasets (Fig.7)
(For detailed specs, please refer [The Discovery Cluster](https://rc-docs.northeastern.edu/en/latest/welcome/welcome.html) 
at [MGHPCC](https://www.mghpcc.org/)):   
- *Processor*: 2.4 GHz Intel E5-2680 v4 CPUs   
- *Memory*: 256 GB/node   
- *Secondary Storage*:  GPFS, disk type unavailable, assume SSD since it’s a world-class HPC facility.    
- *Network*: InfiniBand (IB) interconnect running at 100 Gbps



### Repeating the Experiments
#### Jupyter notebooks 
You can repeat the experiments and produce all figures from the paper by using two Jupyter notebooks,
one for synthetic data and the other for real datasets: 

* [`/experiments_sigmod20/Figures_syntheticdata_sigmod20.ipynb`](https://github.com/northeastern-datalab/factorized-graphs/blob/master/experiments_sigmod20/Figures_syntheticdata_sigmod20.ipynb): 
 all experiments with synthetic data sets
* [`/experiments_sigmod20/Figures_realdata_sigmod20.ipynb`](https://github.com/northeastern-datalab/factorized-graphs/blob/master/experiments_sigmod20/Figures_realdata_sigmod20.ipynb):
 all experiments on real-world datasets
 
Run the Jupyter notebook and jump to the necessary functions to learn how a certain figure was generated in the paper.

#### Cached experimental traces


The code allows two levels of granularity to reproduce all results:
  1. Plot figures from the paper **using our saved experimental traces**: We have cached all the intermediate results 
  required to produce the figures in our paper in the 
  [`/experiments_sigmod20/datacache`](https://github.com/northeastern-datalab/factorized-graphs/blob/master/experiments_sigmod20/datacache) subfolder. 
  You can simply run the cells in Jupyter notebook to generate all figures. 
  By default the code uses our stored results and we recommend this mode during the first pass.
  2. In order to **run all the experiments from scratch**, please use the option `create_data =  True` in  the provided Jupyter notebooks. 
  On a 2016 Macbook Pro, this would  take approximately ###$$$ hours for the synthetic, and ###$$$ hours for the real data experiments. 
  We suggest to thus reproduce the graphs with less accuracy instead (fewer data samples and thus more wiggly) by using changing the parameters ###$$$ in the notebook

**Example Use**: 
To generate Fig 5(a) using cached intermediate data, run   
  `Fig_Backtracking_Advantage.run(choice=31, variant=0, create_data=False, show_plot=True, create_pdf=True)`   
To ignore the cached data and recreate figures from scratch  
  `Fig_Backtracking_Advantage.run(choice=31, variant=0, create_data=True, show_plot=True, create_pdf=True)`  

Please note that `choice` and `variant` parameter values are already set to the ones used for the paper.   

**A note about timing**:
Many of the accuracy experiments over real datasets (Figure.7) were run at [MGHPCC](https://www.mghpcc.org/) 
on high-performance compute infrastructure. Although the plots are reproducible instantaneously from cached data provided in the repository, 
if you wish to recreate the experimental data points from scratch (i.e. using the `create_data=True` flag), 
it is highly recommended to run the real data experiments with a large number of CPU cores and plenty of memory. 
They are feasible to run on a home computer, but it will likely take multiple days to produce plots with comparable variance 
to those presented in the paper.
