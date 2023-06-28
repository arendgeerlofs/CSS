# Investigating Self Organised Criticality (SOC) properties of the human brain
Complex System Simulation course 2023 - Group 3 

This is the implementation of our project for the course of Complex Systems Simulation. 
In this project, we aim to identify the necessary and sufficient conditions for SOC to occur in the human brain.

Below is an example of spiking activity of neurons in a simulation. The neurons are organised in a 50x50 square lattice. A neuron spikes when its Voltage membrane surpasses the defined threshold. 
In this simulation of the system, where there is a slow drive (h>0), one can observe the spatial propagation of spikes during successive time frames. 
![driven_avalanche_simulation](https://github.com/arendgeerlofs/CSS/assets/113594011/d2b5da60-7f12-4510-ba7b-6f978052a5e2)


## Requirements
In order to run the simulation properly, please make sure to install the following package requirements by `pip install -r requirements.txt` .

* numpy
* numba
* matplotlib
* pandas
* scipy
* tqdm

## Simulation code & Experiments
We have implemented two main models, the Bak-Tang-Wiesenfeld (BTW) and a stochastic Branching model, in order to make the aforementioned investigation. 

To obtain results from the BTW model, run the `Experiments_BTW` file, which performs several experiments. 
In order to make the simulations shorter in time, you can change the parameter configurations, namely choose a value for `frames` < 35000 or a smaller grid `size` < 50.


To obtain results from the Stochastic Branching model, run the `Stochastic_Branching_experiments` file.

One can also find some results from the simulations in the "results" folder of this repository. 

