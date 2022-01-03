# Citation bias study

## ISSI_2021  
This folder contains a jupyter notebook for the calculation contained in the work-in-progress paper submitted to ISSI 2021  

## Jasmine  
This folder contains Jasmine Yuan's work  

## NetworkX_migration  
This folder contains files for the iGraph to NetworkX code migration  
March 18, 2021: made a new file called "networkx-main" for Jasmine to update every week with her code

## NetworkSimulation
This folder contains codes that simulates citation networks with varied degrees of bias.

network_generation_toy_model.py simulates a network where a paper X published in year Y has a pre-defined probability p (derived from the observed network) to cite a paper Z published in prior years. A random number is drawn, if the random number is smaller than or equal to to the probability p, we will assign a citation relationship from X to Z.
