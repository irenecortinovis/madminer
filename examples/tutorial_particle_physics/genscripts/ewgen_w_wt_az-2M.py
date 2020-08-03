
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import matplotlib
from matplotlib import pyplot as plt


from madminer.core import MadMiner
from madminer.delphes import DelphesReader
from madminer.sampling import combine_and_shuffle
from madminer.plotting import plot_distributions


# In[2]:


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



miner = MadMiner()
miner.load("/data_CMS/cms/cortinovis/ewdim6/data_ew_2M_az/setup.h5")

miner.run(
    sample_benchmark='sm',
    temp_directory = './',
    mg_directory=mg_dir,
    mg_process_directory='/data_CMS/cms/cortinovis/ewdim6/mg_processes_ew_2M_az/signal_pythia',
    proc_card_file='cards/proc_card_signal.dat',
    param_card_template_file='cards/param_card_template.dat',
    pythia8_card_file='cards/pythia8_card.dat',
    run_card_file='cards/run_card_signal_large.dat',
    log_directory='/data_CMS/cms/cortinovis/ewdim6/logs_ew_2M_az/signal',
    initial_command="PYTHONPATH=/usr/lib64/python",
    python2_override=True,
)


additional_benchmarks = ['w', 'ww', 'neg_w', 'neg_ww']

miner.run_multiple(
    sample_benchmarks=additional_benchmarks,
    mg_directory=mg_dir,
    temp_directory = './',
    mg_process_directory='/data_CMS/cms/cortinovis/ewdim6/mg_processes_ew_2M_az/signal_pythia2',
    proc_card_file='cards/proc_card_signal.dat',
    param_card_template_file='cards/param_card_template.dat',
    pythia8_card_file='cards/pythia8_card.dat',
    run_card_files=['cards/run_card_signal_small.dat'],
    log_directory='/data_CMS/cms/cortinovis/ewdim6/logs_ew_2M_az/signal2',
    initial_command="PYTHONPATH=/usr/lib64/python",
    python2_override=True,
)



delphes = DelphesReader('/data_CMS/cms/cortinovis/ewdim6/data_ew_2M_az/setup.h5')

delphes.add_sample(
    lhe_filename='/data_CMS/cms/cortinovis/ewdim6/mg_processes_ew_2M_az/signal_pythia/Events/run_01/unweighted_events.lhe.gz',
    hepmc_filename='/data_CMS/cms/cortinovis/ewdim6/mg_processes_ew_2M_az/signal_pythia/Events/run_01/tag_1_pythia8_events.hepmc.gz',
    sampled_from_benchmark='sm',
    is_background=False,
    k_factor=1,
)

for i, benchmark in enumerate(additional_benchmarks):
    delphes.add_sample(
        lhe_filename='/data_CMS/cms/cortinovis/ewdim6/mg_processes_ew_2M_az/signal_pythia2/Events/run_0{}/unweighted_events.lhe.gz'.format(i+1),
        hepmc_filename='/data_CMS/cms/cortinovis/ewdim6/mg_processes_ew_2M_az/signal_pythia2/Events/run_0{}/tag_1_pythia8_events.hepmc.gz'.format(i+1),
        sampled_from_benchmark=benchmark,
        is_background=False,
        k_factor=1,
    )


delphes.run_delphes(
    delphes_directory=mg_dir + '/HEPTools/Delphes-3.4.2',
    delphes_card='cards/delphes_card_CMS.dat',
    log_file='/data_CMS/cms/cortinovis/ewdim6/logs_ew_2M_az/delphes.log',
)



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
    'eta_j1',
    'j[0].eta',
    required=False,
    default=float('nan'),
    #default=0,
)

delphes.add_observable(
    'eta_j2',
    'j[1].eta',
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
    'H_eta',
    '(lep1ZZ+lep2ZZ+lep3ZZ+lep4ZZ).eta',
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
    'Z2_m',
    '(lep3ZZ+lep4ZZ).m',
    required=True,
)

delphes.add_observable(
    'Z1_eta',
    '(lep1ZZ+lep2ZZ).eta',
    required=True,
)

delphes.add_observable(
    'Z2_eta',
    '(lep3ZZ+lep4ZZ).eta',
    required=True,
)

delphes.add_observable(
    'delta_phi_zz',
    '(lep1ZZ+lep2ZZ).deltaphi(lep3ZZ+lep4ZZ) * (-1. + 2.*float((lep1ZZ+lep2ZZ).eta > (lep3ZZ+lep4ZZ).eta))',
    required=False,
    default=float('nan'),
    #default=0,
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


delphes.add_cut('isZZcand == 1')
delphes.add_cut('n_jets >= 2')
delphes.add_cut('m_jj > 700.')

delphes.analyse_delphes_samples()

delphes.save('/data_CMS/cms/cortinovis/ewdim6/data_ew_2M_az/delphes_data.h5')

combine_and_shuffle(
    ['/data_CMS/cms/cortinovis/ewdim6/data_ew_2M_az/delphes_data.h5'],
    '/data_CMS/cms/cortinovis/ewdim6/data_ew_2M_az/delphes_data_shuffled.h5'
)

