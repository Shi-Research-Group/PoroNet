# PoroNet
PoroNet is an inherently interpretable pore graph neural network for predicting gas adsorption capacity of metal-organic frameworks (MOFs). Its interpretability allows the extraction of contributions of individual pores to the total adsorption. PoroNet is constructed on top of pore graphs, which can be generated using code in this repo. The pore graph generation and pore-level energy histogram features will be later incorporated into an open-source Python package, Mofography (unpublished). Stay tuned! <br>

![PoroNet](workflow.jpg)

## Instructions
This repository consists of:
1. "poronet_functions": Functions used in PoroNet.
2. "Download_Tobacco_Database": Download MOF structures from the Tobacco database through MOFXDB (mof.tech.northwestern.edu) database.
3. "Pore_Graph_Generation": Generate pore graphs for chosen MOFs.
4. "Pore_Labels_Extraction": Extract pore-level labels from GCMC simulation trajectories using Mofography.
5. "PoreNet&PoroNet-Base": Illustrations of PoroNet and PoroNet-Base for predicting H2 adsorption at 160 K/5 bar.
6. "Cavity_Size_Distribution_Calculation": Calculation of the cavity size distribution .
7. "Pore_Surface_Area_Calculation":Calcultion of the surface area of pores.
