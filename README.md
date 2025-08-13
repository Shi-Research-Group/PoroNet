# PoroNet
PoroNet is an inherently interpretable pore graph neural network used to predict gas adsorption capacity of metal-organic frameworks (MOF) and extract the contribution of each individual pore to the total adsorption. It was constrcuted on top of pore graphs, which were made using our in-house pore network analysis software "Mofography". Mofography has not been published yet, but the functions related to PoroNet are included in this repository.

This repository consists of:
1. "mofography": Selected functions from Mofography that are involved in PoroNet.
2. "Download_Tobacco_Database": Download MOF structures from the Tobacco databese through MOFXDB (mof.tech.northwestern.edu) database.
3. "Pore_Graph_Generation": Generate pore graphs for chosen MOFs.
4. "Pore_Labels_Extraction": Extract pore-level labels from GCMC simulation trajectories using Mofography.
5. "PoreNet&PoroNet-Base": Illustrations of PoroNet and PoroNet-Base for predicting H2 adsorption at 160 K/5 bar.
