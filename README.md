## MadMiner: Machine learning–based inference for particle physics

*By Johann Brehmer, Felix Kling, Irina Espejo, and Kyle Cranmer*

https://github.com/diana-hep/madminer



## Organization of files:
`examples/tutorial_particle_physics/genscripts`: python scripts following notebook 2b, to generate data <br>
`examples/tutorial_particle_physics/cards`: all the cards needed for the generation of data and delphes <br>
`examples/tutorial_particle_physics/`: jupyter notebooks following the MadMiner tutorial <br>

Different MadGraph models in filenames:
`ew`: ewdim6 <br>
`sme`: smeftsim <br>
`dim6`: custom made (MadMiner) <br>

### Material used in the final thesis:
generation of data: `genscripts/ewgen_w_wt_az-1M.py` (adjust run card) <br>
ALICES training: `3a_likelihood_ratio_ew_w_wt_az-1M.ipynb` <br>
SALLY training: `3b_score_ew_w_wt_az-1M.ipynb` <br>
histogram method and final limits (Asimov SM and simulated BSM signal): `4a_limits_ew_w_wt_az-1M.ipynb` <br>
information geometry at SM (also for other datasets): `4b_fisher_information.ipynb` <br>

### Other material, not used in the final thesis:
`4c_information_geometry.ipynb`: global information geometry for the same dataset used in the final thesis <br>
`ew_w_wt_az-2M`: everything as before, but with 2M: same settings as the dataset for the thesis, but with double MonteCarlo statistics (2 million events) <br>
`ew_*`, `sme_*`, `dim6_*`: Other EFT operators involved, with `ewdim6` model or the other MadGraph models (in particular, `ew_w_wt_02` has the NP activated at the Higgs decay vertex).

