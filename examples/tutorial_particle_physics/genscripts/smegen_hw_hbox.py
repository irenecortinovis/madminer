#MadMiner particle physics tutorial
# 
# # Part 2b: Analyzing events at Delphes level
# 
# Johann Brehmer, Felix Kling, Irina Espejo, and Kyle Cranmer 2018-2019

# In this second part of the tutorial, we'll generate events and extract the observables and weights from them. You have two options: In this notebook we'll do this with Delphes, in the alternative part 2a we stick to parton level.

# ## 0. Preparations

# Before you execute this notebook, make sure you have working installations of MadGraph, Pythia, and Delphes.

# In[252]:


from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import matplotlib
from matplotlib import pyplot as plt


from madminer.core import MadMiner
from madminer.delphes import DelphesReader
from madminer.sampling import combine_and_shuffle
from madminer.plotting import plot_distributions


# In[247]:


# MadMiner output
logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.DEBUG
)

# Output of all other modules (e.g. matplotlib)
for key in logging.Logger.manager.loggerDict:
    if "madminer" not in key:
        logging.getLogger(key).setLevel(logging.WARNING)


mg_dir = '/home/llr/cms/cortinovis/EFT2Obs/MG5_aMC_v2_6_7'

# Please enter here the path to your MG5 root directory. This notebook assumes that you installed Delphes and Pythia through MG5.

# In[248]:


# ## 1. Generate events

# Let's load our setup:

# In[250]:


miner = MadMiner()
miner.load("data_sme_hw_hbox/setup.h5")



# In a next step, MadMiner starts MadGraph and Pythia to generate events and calculate the weights. You can use `run()` or `run_multiple()`; the latter allows to generate different runs with different run cards and optimizing the phase space for different benchmark points. 
# 
# In either case, you have to provide paths to the process card, run card, param card (the entries corresponding to the parameters of interest will be automatically adapted), and an empty reweight card. Log files in the `log_directory` folder collect the MadGraph output and are important for debugging.
# 
# The `sample_benchmark` (or in the case of `run_all`, `sample_benchmarks`) option can be used to specify which benchmark should be used for sampling, i.e. for which benchmark point the phase space is optimized. If you just use one benchmark, reweighting to far-away points in parameter space can lead to large event weights and thus large statistical fluctuations. It is therefore often a good idea to combine at least a few different benchmarks for this option. Here we use the SM and the benchmark "w" that we defined during the setup step.
# 
# One slight annoyance is that MadGraph only supports Python 2. The `run()` and `run_multiple()` commands have a keyword `initial_command` that let you load a virtual environment in which `python` maps to Python 2 (which is what we do below). Alternatively / additionally you can set `python2_override=True`, which calls `python2.7` instead of `python` to start MadGraph.

# In[210]:


miner.run(
    sample_benchmark='sm',
    temp_directory = './',
    mg_directory=mg_dir,
    mg_process_directory='./mg_processes_sme_hw_hbox/signal_pythia',
    proc_card_file='cards/proc_card_signal_hzz4l_vbf.dat',
    param_card_template_file='cards/param_card_template_smeftsim_u35_massless_chw_cph2.dat',
    pythia8_card_file='cards/pythia8_card.dat',
    run_card_file='cards/run_card_signal_large.dat',
    log_directory='logs_sme_hw_hbox/signal',
    initial_command="PYTHONPATH=/usr/lib64/python",
    python2_override=True,
)


# In[231]:


additional_benchmarks = ['w', 'ww', 'neg_w', 'neg_ww']


# In[212]:


miner.run_multiple(
    sample_benchmarks=additional_benchmarks,
    mg_directory=mg_dir,
    temp_directory = './',
    mg_process_directory='./mg_processes_sme_hw_hbox/signal_pythia2',
    proc_card_file='cards/proc_card_signal_hzz4l_vbf.dat',
    param_card_template_file='cards/param_card_template_smeftsim_u35_massless_chw_cph2.dat',
    pythia8_card_file='cards/pythia8_card.dat',
    run_card_files=['cards/run_card_signal_small.dat'],
    log_directory='logs_sme_hw_hbox/signal2',
    initial_command="PYTHONPATH=/usr/lib64/python",
    python2_override=True,
)



# Finally, note that both `MadMiner.run()` and `MadMiner.run_multiple()` have a `only_create_script` keyword. If that is set to True, MadMiner will not start the event generation directly, but prepare folders with all the right settings and ready-to-run bash scripts. This might make it much easier to generate Events on a high-performance computing system. 

# ## 2. Run Delphes

# The `madminer.delphes.DelphesReader` class wraps around Delphes, a popular fast detector simulation, to simulate the effects of the detector.

# In[232]:


delphes = DelphesReader('data_sme_hw_hbox/setup.h5')


# After creating the `DelphesReader` object, one can add a number of event samples (the output of running MadGraph and Pythia in step 1 above) with the `add_sample()` function.
# 
# In addition, you have to provide the information which sample was generated from which benchmark with the `sampled_from_benchmark` keyword, and set `is_background=True` for all background samples.

# In[233]:


delphes.add_sample(
    lhe_filename='mg_processes_sme_hw_hbox/signal_pythia/Events/run_01/unweighted_events.lhe.gz',
    hepmc_filename='mg_processes_sme_hw_hbox/signal_pythia/Events/run_01/tag_1_pythia8_events.hepmc.gz',
    sampled_from_benchmark='sm',
    is_background=False,
    k_factor=1.1, #Why??
)

for i, benchmark in enumerate(additional_benchmarks):
    delphes.add_sample(
        lhe_filename='mg_processes_sme_hw_hbox/signal_pythia2/Events/run_0{}/unweighted_events.lhe.gz'.format(i+1),
        hepmc_filename='mg_processes_sme_hw_hbox/signal_pythia2/Events/run_0{}/tag_1_pythia8_events.hepmc.gz'.format(i+1),
        sampled_from_benchmark=benchmark,
        is_background=False,
        k_factor=1.1,
    )

"""
delphes.add_sample(
    lhe_filename='mg_processes/background_pythia/Events/run_01/unweighted_events.lhe.gz',
    hepmc_filename='mg_processes/background_pythia/Events/run_01/tag_1_pythia8_events.hepmc.gz',
    sampled_from_benchmark='sm',
    is_background=True,
    k_factor=1.0,
"""


# Now we run Delphes on these samples (you can also do this externally and then add the keyword `delphes_filename` when calling `DelphesReader.add_sample()`):

# In[234]:


delphes.run_delphes(
    delphes_directory=mg_dir + '/HEPTools/Delphes-3.4.2',
    delphes_card='cards/delphes_card_CMS.dat',
    log_file='logs_sme_hw_hbox/delphes.log',
)


# ## 3. Observables and cuts

# The next step is the definition of observables, either through a Python function or an expression that can be evaluated. Here we demonstrate the latter, which is implemented in `add_observable()`. In the expression string, you can use the terms `j[i]`, `e[i]`, `mu[i]`, `a[i]`, `met`, where the indices `i` refer to a ordering by the transverse momentum. In addition, you can use `p[i]`, which denotes the `i`-th particle in the order given in the LHE sample (which is the order in which the final-state particles where defined in MadGraph).
# 
# All of these represent objects inheriting from scikit-hep [LorentzVectors](http://scikit-hep.org/api/math.html#vector-classes), see the link for a documentation of their properties. In addition, they have `charge` and `pdg_id` properties.
# 
# `add_observable()` has an optional keyword `required`. If `required=True`, we will only keep events where the observable can be parsed, i.e. all involved particles have been detected. If `required=False`, un-parseable observables will be filled with the value of another keyword `default`.
# 
# In a realistic project, you would want to add a large number of observables that capture all information in your events. Here we will just define two observables, the transverse momentum of the leading (= higher-pT) jet, and the azimuthal angle between the two leading jets.

# In[235]:


delphes.add_observable(
    'pt_j1',
    'j[0].pt',
    required=False,
    default=float('nan'),
    #default=0,
)

delphes.add_observable(
    'delta_phi_jj',
    'j[0].deltaphi(j[1]) * (-1. + 2.*float(j[0].eta > j[1].eta))',
    required=False,
    default=float('nan'),
    #default=0,
)

delphes.add_observable(
    'delta_eta_jj',
    'j[0].deltaeta(j[1]) * (-1. + 2.*float(j[0].eta > j[1].eta))',
    required=False,
    default=float('nan'),
    #default=0,
)
    

delphes.add_observable(
    'm_jj',
    '(j[0]+j[1]).m',
    required=False,
    default=float('nan'),
    #default=0,
)
    
delphes.add_observable(
    'isZZcand',
    'isZZcand',
    required=True,
)

delphes.add_observable(
    'H_pt',
    '(lep1ZZ+lep2ZZ+lep3ZZ+lep4ZZ).pt',
    required=True,
)

delphes.add_observable(
    'H_m',
    '(lep1ZZ+lep2ZZ+lep3ZZ+lep4ZZ).m',
    required=True,
)

delphes.add_observable(
    'Z1_pt',
    '(lep1ZZ+lep2ZZ).pt',
    required=True,
)

delphes.add_observable(
    'Z2_pt',
    '(lep3ZZ+lep4ZZ).pt',
    required=True,
)

delphes.add_observable(
    'H_eta',
    '(lep1ZZ+lep2ZZ+lep3ZZ+lep4ZZ).eta',
    required=True,
)

delphes.add_observable(
    'n_jets',
    'len(j)',
    required=True,
)

delphes.add_observable(
    'met',
    'met.pt',
    required=True,
)


# We can also add cuts, again in parse-able strings. In addition to the objects discussed above, they can contain the observables:

# In[236]:


delphes.add_cut('isZZcand == 1')
delphes.add_cut('n_jets >= 2')


# ## 4. Analyse events and store data

# The function `analyse_samples` then calculates all observables from the Delphes file(s) generated before and checks which events pass the cuts:

# In[237]:


delphes.analyse_delphes_samples()


# In[238]:


delphes.save('data_sme_hw_hbox/delphes_data.h5')


combine_and_shuffle(
    ['data_sme_hw_hbox/delphes_data.h5'],
    'data_sme_hw_hbox/delphes_data_shuffled.h5'
)




