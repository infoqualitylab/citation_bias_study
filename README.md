# Citation bias study

## ISSI_2021  
This folder contains a jupyter notebook for the calculation contained in the work-in-progress paper submitted to ISSI 2021  
Yuanxi Fu+, Jasmine Yuan+, Jodi Schneider. “Using Citation Bias to Guide Better Sampling of Scientific Literature.” [work in progress paper] In Proceedings of the 18th International Conference on Scientometrics & Informetrics (ISSI2021), Virtual, July 2021. 419–424.

## Jasmine  
This folder contains Jasmine Yuan's work  

## NetworkX_migration  
This folder contains files for the iGraph to NetworkX code migration  
March 18, 2021: made a new file called "networkx-main" for Jasmine to update every week with her code

## NetworkSimulation
This folder contains codes that simulates citation networks with varied degrees of bias.

network_generation_toy_model.py simulates a network where a paper X published in year Y has a pre-defined probability p (derived from the observed network) to cite a paper Z published in prior years. A random number is drawn, if the random number is smaller than or equal to to the probability p, we will assign a citation relationship from X to Z.

### Observed Network
![Observed Network](https://github.com/infoqualitylab/citation_bias_study/blob/master/NetworkSimulation/observed_network.png)

Data from: de Vries, Y. A., & Munafo, M. (2016). [Dataset] Citation bias and selective focus on positive findings in the literature on 5-HTTLPR, life stress, and depression. University of Bristol. Retrieved January 28, 2021, from http://doi.org/10.5523/BRIS.Z7JCONXFBMDR1JJ3T0W4K1HWN 

### Simulated Network via an equal probability (no bias) Toy Model
![Simulated Network](https://github.com/infoqualitylab/citation_bias_study/blob/master/NetworkSimulation/simulated_network.png)

## Zhonghe  
This folder contains Zhonghe's work  

## ResearchShowcase
This folder contains codes that are used to produce the work in the iSchool Research Showcase poster:
Zhonghe Wan, Yuanxi Fu, Jodi Schneider. Using citation redistribution to estimate unbiased expected citation count from a biased citation network. Poster presented at the iSchool Research Showcase, University of Illinois at Urbana Champaign, October 27, 2021. http://hdl.handle.net/2142/112794



