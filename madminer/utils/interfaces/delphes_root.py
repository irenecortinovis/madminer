from __future__ import absolute_import, division, print_function, unicode_literals
import six

import numpy as np
from collections import OrderedDict
import uproot
import os
import logging

from madminer.utils.particle import MadMinerParticle
from madminer.utils.various import math_commands

logger = logging.getLogger(__name__)


def parse_delphes_root_file(
    delphes_sample_file,
    observables,
    observables_required,
    observables_defaults,
    cuts,
    cuts_default_pass,
    weight_labels=None,
    use_generator_truth=False,
    acceptance_pt_min_e=None,
    acceptance_pt_min_mu=None,
    acceptance_pt_min_a=None,
    acceptance_pt_min_j=None,
    acceptance_eta_max_e=None,
    acceptance_eta_max_mu=None,
    acceptance_eta_max_a=None,
    acceptance_eta_max_j=None,
    delete_delphes_sample_file=False,
):
    """ Extracts observables and weights from a Delphes ROOT file """

    logger.debug("Parsing Delphes file %s", delphes_sample_file)
    if weight_labels is None:
        logger.debug("Not extracting weights")
    else:
        logger.debug("Extracting weights %s", weight_labels)

    # Delphes ROOT file
    root_file = uproot.open(str(delphes_sample_file))
    # The str() call is important when using numpy 1.16.0 and Python 2.7. In this combination of versions, a unicode
    # delphes_sample_file would lead to a crash.

    # Delphes tree
    tree = root_file["Delphes"]

    # Weights
    n_weights = 0
    weights = None
    if weight_labels is not None:
        try:
            weights = tree.array("Weight.Weight")

            n_weights = len(weights[0])
            n_events = len(weights)

            logger.debug("Found %s events, %s weights", n_events, n_weights)

            weights = np.array(weights).reshape((n_events, n_weights)).T
        except KeyError:
            raise RuntimeError(
                "Extracting weights from Delphes ROOT file failed. Please install inofficial patches"
                " for the MG-Pythia interface and Delphes, available upong request, or parse weights"
                " from the LHE file!"
            )
    else:
        n_events = _get_n_events(tree)
        logger.debug("Found %s events", n_events)

    # Get all particle properties
    if use_generator_truth:
        photons_all_events = _get_particles_truth(tree, acceptance_pt_min_a, acceptance_eta_max_a, [22])
        electrons_all_events = _get_particles_truth(tree, acceptance_pt_min_a, acceptance_eta_max_a, [11, -11])
        muons_all_events = _get_particles_truth(tree, acceptance_pt_min_a, acceptance_eta_max_a, [13, -13])
        leptons_all_events = _get_particles_truth_leptons(
            tree, acceptance_pt_min_e, acceptance_eta_max_e, acceptance_pt_min_mu, acceptance_eta_max_mu
        )
        jets_all_events = _get_particles_truth_jets(tree, acceptance_pt_min_j, acceptance_eta_max_j)
        met_all_events = _get_particles_truth_met(tree)

    else:
        photons_all_events = _get_particles_photons(tree, acceptance_pt_min_a, acceptance_eta_max_a)
        electrons_all_events = _get_particles_charged(
            tree, "Electron", 0.000511, -11, acceptance_pt_min_e, acceptance_eta_max_e
        )
        muons_all_events = _get_particles_charged(tree, "Muon", 0.105, -13, acceptance_pt_min_mu, acceptance_eta_max_mu)
        leptons_all_events = _get_particles_leptons(
            tree, acceptance_pt_min_e, acceptance_eta_max_e, acceptance_pt_min_mu, acceptance_eta_max_mu
        )
        jets_all_events = _get_particles_jets(tree, acceptance_pt_min_j, acceptance_eta_max_j)
        met_all_events = _get_particles_met(tree)

    # Prepare variables
    def get_objects(ievent):


        visible_momentum = MadMinerParticle()
        for p in (
            electrons_all_events[ievent]
            + jets_all_events[ievent]
            + muons_all_events[ievent]
            + photons_all_events[ievent]
        ):
            visible_momentum += p
        all_momentum = visible_momentum + met_all_events[ievent][0]

        objects = math_commands()
        objects.update(
            {
                "e": electrons_all_events[ievent],
                "j": jets_all_events[ievent],
                "a": photons_all_events[ievent],
                "mu": muons_all_events[ievent],
                "l": leptons_all_events[ievent],
                "met": met_all_events[ievent][0],
                "visible": visible_momentum,
                "all": all_momentum,
                "boost_to_com": lambda momentum: momentum.boost(all_momentum.boost_vector()),
            }
        )

        #find same flavour opposite sign lepton pairs,
        #given the list of electrons or muon in an event
        def find_SFOS(cand_list, part_list, idxlists=False):
            #positively and negatively charged leptons
            cand_pos = []
            cand_neg = []
            for idx, (is_cand, part) in enumerate(zip(cand_list, part_list)):
                if is_cand != 0:
                    if part.charge > 0:
                        cand_pos.append(idx)
                    if part.charge < 0:
                        cand_neg.append(idx)
            #the max number of SFOS pairs is
            nmax_sfos = min(len(cand_pos), len(cand_neg))
            if idxlists == False:
                return nmax_sfos
            else:
                return nmax_sfos, cand_pos, cand_neg


        def check_ZZ_cand(cand_e, part_e, cand_m, part_m, nSFOS):
            #if at least 2 sfos
            if find_SFOS(cand_e, part_e) + find_SFOS(cand_m, part_m) >= nSFOS:
                return True
            else:
                return False

        candidate_es = np.ones(len(objects["e"]))
        candidate_mus = np.ones(len(objects["mu"]))
        isZZcand = 0
        isZ1e = isZ2e = -99
        idxl1 = idxl2 = idxl3 = idxl4 = -99
        lep1 = lep2 = lep3 = lep4 = 0

        #find candidate leptons
        if check_ZZ_cand(candidate_es, objects["e"], candidate_mus, objects["mu"], nSFOS=2) == True:
            #if eta and pt cuts
            for idx, el in enumerate(objects["e"]):
                if not (abs(el.eta) < 2.5 and el.pt > 7):
                    candidate_es[idx] = 0
            for idx, mu in enumerate(objects["mu"]):
                if not (abs(mu.eta) < 2.4 and mu.pt > 5):
                    candidate_mus[idx] = 0

        #find Z candidates
        if check_ZZ_cand(candidate_es, objects["e"], candidate_mus, objects["mu"], nSFOS=2) == True:
            #Z1
            idx_m_mu = -1
            idx_m_e = -1
            #electrons
            nmax_e_sfos, cand_e_pos, cand_e_neg = find_SFOS(candidate_es, objects["e"], idxlists=True)
            short_idx_cand_e = min([cand_e_pos, cand_e_neg])
            long_idx_cand_e = max([cand_e_pos, cand_e_neg])

            possible_z1_e = []
            for idx1 in short_idx_cand_e:
                el1 = (objects["e"])[idx1]
                for idx2 in long_idx_cand_e:
                    el2 = (objects["e"])[idx2]
                    #calculate stuff
                    mz1 = (el1+el2).m
                    if(((el1.pt > 20 and el2.pt > 10) or (el2.pt > 20 and el1.pt > 10))
                        and mz1 > 60):
                        possible_z1_e.append([idx1, idx2, mz1])

            m_poss_z1_e = [item[2] for item in possible_z1_e]
            best_m_e = min(m_poss_z1_e, key=lambda x:abs(x-91.2), default=-1)
            if best_m_e != -1:
                idx_m_e = m_poss_z1_e.index(best_m_e)

            #muons
            nmax_mu_sfos, cand_mu_pos, cand_mu_neg = find_SFOS(candidate_mus, objects["mu"], idxlists=True)
            short_idx_cand_mu = min([cand_mu_pos, cand_mu_neg])
            long_idx_cand_mu = max([cand_mu_pos, cand_mu_neg])

            possible_z1_mu = []
            for idx1 in short_idx_cand_mu:
                mu1 = (objects["mu"])[idx1]
                for idx2 in long_idx_cand_mu:
                    mu2 = (objects["mu"])[idx2]
                    #calculate stuff
                    mz1 = (mu1+mu2).m
                    if(((mu1.pt > 20 and mu2.pt > 10) or (mu2.pt > 20 and mu1.pt > 10))
                        and mz1 > 60):
                        possible_z1_mu.append([idx1, idx2, mz1])

            m_poss_z1_mu = [item[2] for item in possible_z1_mu]
            best_m_mu = min(m_poss_z1_mu, key=lambda x:abs(x-91.2), default=-1)
            if best_m_mu != -1:
                idx_m_mu = m_poss_z1_mu.index(best_m_mu)

            #choose best z1 candidate (mass closest to Z)
            #save the z1 candidates indices
            #set to 0 the chosen z1 candidates, so they will not be available for z2

            if (idx_m_e != -1) or (idx_m_mu != -1):
                #if candidate SFOS is mu+mu-
                if(abs(best_m_mu-91.2) < abs(best_m_e-91.2)):
                    isZ1e = 0
                    idxl1 = possible_z1_mu[idx_m_mu][0]
                    idxl2 = possible_z1_mu[idx_m_mu][1]
                    candidate_mus[idxl1] = 0
                    candidate_mus[idxl2] = 0
                    lep1 = (objects["mu"])[idxl1]
                    lep2 = (objects["mu"])[idxl2]
                #if candidate SFOS is e+e-
                else:
                    isZ1e = 1
                    idxl1 = possible_z1_e[idx_m_e][0]
                    idxl2 = possible_z1_e[idx_m_e][1]
                    candidate_es[idxl1] = 0
                    candidate_es[idxl2] = 0
                    lep1 = (objects["e"])[idxl1]
                    lep2 = (objects["e"])[idxl2]


            if idxl1 != -99 and idxl2 != -99 and check_ZZ_cand(candidate_es, objects["e"], candidate_mus, objects["mu"], nSFOS=1) == True:
                #Z2
                idx_pt_e = -1
                idx_pt_mu = -1
                #electrons
                nmax_e_sfos, cand_e_pos, cand_e_neg = find_SFOS(candidate_es, objects["e"], idxlists=True)
                short_idx_cand_e = min([cand_e_pos, cand_e_neg])
                long_idx_cand_e = max([cand_e_pos, cand_e_neg])

                possible_z2_e = []

                for idx3 in short_idx_cand_e:
                    el3 = (objects["e"])[idx3]
                    for idx4 in long_idx_cand_e:
                        el4 = (objects["e"])[idx4]
                        #calculate stuff
                        mz2 = (el3+el4).m
                        mzz = (lep1+lep1+el3+el4).m
                        if(mz2 > 12 and mzz > 100 and mzz < 150):
                            possible_z2_e.append([idx3, idx4, el3.pt+el4.pt])

                #choose z2 from electrons with highest pt
                pt_poss_z2_e = [item[2] for item in possible_z2_e]
                best_pt_e = max(pt_poss_z2_e, default=-1)
                if best_pt_e != -1:
                    idx_pt_e = pt_poss_z2_e.index(best_pt_e)


                #muons
                nmax_mu_sfos, cand_mu_pos, cand_mu_neg = find_SFOS(candidate_mus, objects["mu"], idxlists=True)
                short_idx_cand_mu = min([cand_mu_pos, cand_mu_neg])
                long_idx_cand_mu = max([cand_mu_pos, cand_mu_neg])

                possible_z2_mu = []

                for idx3 in short_idx_cand_mu:
                    mu3 = (objects["mu"])[idx3]
                    for idx4 in long_idx_cand_mu:
                        mu4 = (objects["mu"])[idx4]
                        #calculate stuff
                        mz2 = (mu3+mu4).m
                        mzz = (lep1+lep1+mu3+mu4).m
                        if(mz2 > 12 and mzz > 100 and mzz < 150):
                            possible_z2_mu.append([idx3, idx4, mu3.pt+mu4.pt])

                #choose z2 from muons with highest pt
                pt_poss_z2_mu = [item[2] for item in possible_z2_mu]
                best_pt_mu = max(pt_poss_z2_mu, default=-1)
                if best_pt_mu != -1:
                    idx_pt_mu = pt_poss_z2_mu.index(best_pt_mu)


                #choose best z2 candidate (highest pt)
                #save the z2 candidates indices
                #set to 0 the chosen z2 candidates, so they will not be available for z2
                if (idx_pt_e != -1) or (idx_pt_mu != -1): 
                    #if candidate SFOS is mu+mu-
                    if(best_pt_mu > best_pt_e):
                        isZ2e = 0
                        idxl3 = possible_z2_mu[idx_pt_mu][0]
                        idxl4 = possible_z2_mu[idx_pt_mu][1]
                        candidate_mus[idxl3] = 0
                        candidate_mus[idxl4] = 0
                        lep3 = (objects["mu"])[idxl3]
                        lep4 = (objects["mu"])[idxl4]
                        isZZcand = 1
                    #if candidate SFOS is e+e-
                    else:
                        isZ2e = 1
                        idxl3 = possible_z2_e[idx_pt_e][0]
                        idxl4 = possible_z2_e[idx_pt_e][1]
                        candidate_es[idxl3] = 0
                        candidate_es[idxl4] = 0
                        lep3 = (objects["e"])[idxl3]
                        lep4 = (objects["e"])[idxl4]
                        isZZcand = 1

        logger.debug("Value of isZZcand: %d", isZZcand)

        #TODO
        #iso cut
        #primary vertex cut
        #mz1,mz2

        #update objects dictionary
        objects.update(
            {
                "isZZcand": isZZcand,
                "lep1ZZ": lep1,
                "lep2ZZ": lep2,
                "lep3ZZ": lep3,
                "lep4ZZ": lep4,
            }
        )

        return objects

    # Observations
    observable_values = OrderedDict()

    for obs_name, obs_definition in six.iteritems(observables):
        values_this_observable = []

        # Loop over events
        for event in range(n_events):
            variables = get_objects(event)

            if isinstance(obs_definition, six.string_types):
                try:
                    values_this_observable.append(eval(obs_definition, variables))
                except (SyntaxError, NameError, TypeError, ZeroDivisionError, IndexError):
                    default = observables_defaults[obs_name]
                    if default is None:
                        default = np.nan
                    values_this_observable.append(default)
            else:
                try:
                    values_this_observable.append(
                        obs_definition(
                            leptons_all_events[event],
                            photons_all_events[event],
                            jets_all_events[event],
                            met_all_events[event][0],
                        )
                    )
                except RuntimeError:
                    default = observables_defaults[obs_name]
                    if default is None:
                        default = np.nan
                    values_this_observable.append(default)

        values_this_observable = np.array(values_this_observable, dtype=np.float)
        observable_values[obs_name] = values_this_observable

        logger.debug("  First 10 values for observableeeeee %s:\n%s", obs_name, values_this_observable[:10])

    # Cuts
    cut_values = []

    for cut, default_pass in zip(cuts, cuts_default_pass):
        values_this_cut = []

        # Loop over events
        for event in range(n_events):
            variables = get_objects(event)

            for obs_name in observable_values:
                variables[obs_name] = observable_values[obs_name][event]

            try:
                values_this_cut.append(eval(cut, variables))
            except (SyntaxError, NameError, TypeError, ZeroDivisionError, IndexError):
                values_this_cut.append(default_pass)

        values_this_cut = np.array(values_this_cut, dtype=np.bool)
        cut_values.append(values_this_cut)

    # Check for existence of required observables
    combined_filter = None

    for obs_name, obs_required in six.iteritems(observables_required):
        if obs_required:
            this_filter = np.isfinite(observable_values[obs_name])
            n_pass = np.sum(this_filter)
            n_fail = np.sum(np.invert(this_filter))

            logger.debug("  %s / %s events pass required observable %s", n_pass, n_pass + n_fail, obs_name)

            if combined_filter is None:
                combined_filter = this_filter
            else:
                combined_filter = np.logical_and(combined_filter, this_filter)

    # Check cuts
    for cut, values_this_cut in zip(cuts, cut_values):
        n_pass = np.sum(values_this_cut)
        n_fail = np.sum(np.invert(values_this_cut))

        logger.debug("  %s / %s events pass cut %s", n_pass, n_pass + n_fail, cut)

        if combined_filter is None:
            combined_filter = values_this_cut
        else:
            combined_filter = np.logical_and(combined_filter, values_this_cut)

    # Apply filter
    if combined_filter is not None:
        n_pass = np.sum(combined_filter)
        n_fail = np.sum(np.invert(combined_filter))

        if n_pass == 0:
            logger.warning("  No observations remainining!")

            return None, None, combined_filter

        logger.info("  %s / %s events pass everything", n_pass, n_pass + n_fail)

        for obs_name in observable_values:
            observable_values[obs_name] = observable_values[obs_name][combined_filter]

        if weights is not None:
            weights = weights[:, combined_filter]

    # Wrap weights
    if weights is None:
        weights_dict = None
    else:
        weights_dict = OrderedDict()
        for weight_label, this_weights in zip(weight_labels, weights):
            weights_dict[weight_label] = this_weights

    # Delete Delphes file
    if delete_delphes_sample_file:
        logger.debug("  Deleting %s", delphes_sample_file)
        os.remove(delphes_sample_file)

    return observable_values, weights_dict, combined_filter


def _get_n_events(tree):
    es = tree.array("Event")
    n_events = len(es)
    return n_events


def _get_particles_truth(tree, pt_min, eta_max, included_pdgids=None):
    es = tree.array("Particle.E")
    pts = tree.array("Particle.PT")
    etas = tree.array("Particle.Eta")
    phis = tree.array("Particle.Phi")
    charges = tree.array("Particle.Charge")
    pdgids = tree.array("Particle.PID")

    all_particles = []

    for ievent in range(len(pts)):
        event_particles = []

        for e, pt, eta, phi, pdgid in zip(es[ievent], pts[ievent], etas[ievent], phis[ievent], pdgids[ievent]):

            if pt_min is not None and pt < pt_min:
                continue
            if eta_max is not None and abs(eta) > eta_max:
                continue
            if (included_pdgids is not None) and (not pdgid in included_pdgids):
                continue

            particle = MadMinerParticle()
            particle.setptetaphie(pt, eta, phi, e)
            particle.set_pdgid(pdgid)
            event_particles.append(particle)

        all_particles.append(event_particles)

    return all_particles


def _get_particles_charged(tree, name, mass, pdgid_positive_charge, pt_min, eta_max):
    pts = tree.array(name + ".PT")
    etas = tree.array(name + ".Eta")
    phis = tree.array(name + ".Phi")
    charges = tree.array(name + ".Charge")

    all_particles = []

    for ievent in range(len(pts)):
        event_particles = []

        for pt, eta, phi, charge in zip(pts[ievent], etas[ievent], phis[ievent], charges[ievent]):

            if pt_min is not None and pt < pt_min:
                continue
            if eta_max is not None and abs(eta) > eta_max:
                continue

            pdgid = pdgid_positive_charge if charge >= 0.0 else -pdgid_positive_charge

            particle = MadMinerParticle()
            particle.setptetaphim(pt, eta, phi, mass)
            particle.set_pdgid(pdgid)
            event_particles.append(particle)

        all_particles.append(event_particles)

    return all_particles


def _get_particles_leptons(tree, pt_min_e, eta_max_e, pt_min_mu, eta_max_mu):
    pt_mu = tree.array("Muon.PT")
    eta_mu = tree.array("Muon.Eta")
    phi_mu = tree.array("Muon.Phi")
    charge_mu = tree.array("Muon.Charge")
    pt_e = tree.array("Electron.PT")
    eta_e = tree.array("Electron.Eta")
    phi_e = tree.array("Electron.Phi")
    charge_e = tree.array("Electron.Charge")

    all_particles = []

    for ievent in range(len(pt_mu)):
        event_particles = []

        # Combined muons and electrons
        event_pts = np.concatenate((pt_mu[ievent], pt_e[ievent]))
        event_etas = np.concatenate((eta_mu[ievent], eta_e[ievent]))
        event_phis = np.concatenate((phi_mu[ievent], phi_e[ievent]))
        event_masses = np.concatenate((0.105 * np.ones_like(pt_mu[ievent]), 0.000511 * np.ones_like(pt_e[ievent])))
        event_charges = np.concatenate((charge_mu[ievent], charge_e[ievent]))
        event_pdgid_positive_charges = np.concatenate(
            (-13 * np.ones_like(pt_mu[ievent], dtype=np.int), -11 * np.ones_like(pt_e[ievent], dtype=np.int))
        )

        # Sort by descending pT
        order = np.argsort(-1.0 * event_pts, axis=None)
        event_pts = event_pts[order]
        event_etas = event_etas[order]
        event_phis = event_phis[order]

        # Create particles
        for pt, eta, phi, mass, charge, pdgid_positive_charge in zip(
            event_pts, event_etas, event_phis, event_masses, event_charges, event_pdgid_positive_charges
        ):

            pdgid = pdgid_positive_charge if charge >= 0.0 else -pdgid_positive_charge

            if abs(int(pdgid)) == 11:
                if pt_min_e is not None and pt < pt_min_e:
                    continue
                if eta_max_e is not None and abs(eta) > eta_max_e:
                    continue

            elif abs(int(pdgid)) == 13:
                if pt_min_mu is not None and pt < pt_min_mu:
                    continue
                if eta_max_mu is not None and abs(eta) > eta_max_mu:
                    continue

            else:
                logger.warning("Delphes ROOT file has lepton with PDG ID %s, ignoring it", pdgid)
                continue

            particle = MadMinerParticle()
            particle.setptetaphim(pt, eta, phi, mass)
            particle.set_pdgid(pdgid)
            event_particles.append(particle)

        all_particles.append(event_particles)

    return all_particles


def _get_particles_truth_leptons(tree, pt_min_e, eta_max_e, pt_min_mu, eta_max_mu):
    es = tree.array("Particle.E")
    pts = tree.array("Particle.PT")
    etas = tree.array("Particle.Eta")
    phis = tree.array("Particle.Phi")
    charges = tree.array("Particle.Charge")
    pdgids = tree.array("Particle.PID")

    all_particles = []

    for ievent in range(len(pts)):
        event_particles = []

        for e, pt, eta, phi, pdgid in zip(es[ievent], pts[ievent], etas[ievent], phis[ievent], pdgids[ievent]):
            if pdgid not in [11, 13, -11, -13]:
                continue
            if pdgid in [11, -11] and (pt_min_e is not None and pt < pt_min_e):
                continue
            if pdgid in [11, -11] and (eta_max_e is not None and abs(eta) > eta_max_e):
                continue
            if pdgid in [13, -13] and (pt_min_mu is not None and pt < pt_min_mu):
                continue
            if pdgid in [13, -13] and (eta_max_mu is not None and abs(eta) > eta_max_mu):
                continue

            particle = MadMinerParticle()
            particle.setptetaphie(pt, eta, phi, e)
            particle.set_pdgid(pdgid)
            event_particles.append(particle)

        all_particles.append(event_particles)

    return all_particles


def _get_particles_photons(tree, pt_min, eta_max):
    pts = tree.array("Photon.PT")
    etas = tree.array("Photon.Eta")
    phis = tree.array("Photon.Phi")
    es = tree.array("Photon.E")

    all_particles = []

    for ievent in range(len(pts)):
        event_particles = []

        for pt, eta, phi, e in zip(pts[ievent], etas[ievent], phis[ievent], es[ievent]):

            if pt_min is not None and pt < pt_min:
                continue
            if eta_max is not None and abs(eta) > eta_max:
                continue

            particle = MadMinerParticle()
            particle.setptetaphie(pt, eta, phi, e)
            particle.set_pdgid(22)
            event_particles.append(particle)

        all_particles.append(event_particles)

    return all_particles


def _get_particles_jets(tree, pt_min, eta_max):
    pts = tree.array("Jet.PT")
    etas = tree.array("Jet.Eta")
    phis = tree.array("Jet.Phi")
    masses = tree.array("Jet.Mass")
    try:
        tau_tags = tree.array("Jet.TauTag")
    except:
        logger.warning("Did not find tau-tag information in Delphes ROOT file.")
        tau_tags = [0 for _ in range(len(pts))]
    try:
        b_tags = tree.array("Jet.BTag")
    except:
        logger.warning("Did not find b-tag information in Delphes ROOT file.")
        b_tags = [0 for _ in range(len(pts))]

    all_particles = []

    for ievent in range(len(pts)):
        event_particles = []

        for pt, eta, phi, mass, tau_tag, b_tag in zip(
            pts[ievent], etas[ievent], phis[ievent], masses[ievent], tau_tags[ievent], b_tags[ievent]
        ):

            if pt_min is not None and pt < pt_min:
                continue
            if eta_max is not None and abs(eta) > eta_max:
                continue

            particle = MadMinerParticle()
            particle.setptetaphim(pt, eta, phi, mass)
            particle.set_pdgid(9)
            particle.set_tags(tau_tag >= 1, b_tag >= 1, False)
            event_particles.append(particle)

        all_particles.append(event_particles)

    return all_particles


def _get_particles_truth_jets(tree, pt_min, eta_max):
    pts = tree.array("GenJet.PT")
    etas = tree.array("GenJet.Eta")
    phis = tree.array("GenJet.Phi")
    masses = tree.array("GenJet.Mass")
    try:
        tau_tags = tree.array("GenJet.TauTag")
    except:
        logger.warning("Did not find tau-tag information for GenJets in Delphes ROOT file.")
        tau_tags = [0 for _ in range(len(pts))]
    try:
        b_tags = tree.array("GenJet.BTag")
    except:
        logger.warning("Did not find b-tag information for GenJets in Delphes ROOT file.")
        b_tags = [0 for _ in range(len(pts))]

    all_particles = []

    for ievent in range(len(pts)):
        event_particles = []

        for pt, eta, phi, mass, tau_tag, b_tag in zip(
            pts[ievent], etas[ievent], phis[ievent], masses[ievent], tau_tags[ievent], b_tags[ievent]
        ):

            if pt_min is not None and pt < pt_min:
                continue
            if eta_max is not None and abs(eta) > eta_max:
                continue

            particle = MadMinerParticle()
            particle.setptetaphim(pt, eta, phi, mass)
            particle.set_pdgid(9)
            particle.set_tags(tau_tag >= 1, b_tag >= 1, False)
            event_particles.append(particle)

        all_particles.append(event_particles)

    return all_particles


def _get_particles_truth_met(tree):
    mets = tree.array("GenMissingET.MET")
    phis = tree.array("GenMissingET.Phi")

    all_particles = []

    for ievent in range(len(mets)):
        event_particles = []

        for met, phi in zip(mets[ievent], phis[ievent]):
            particle = MadMinerParticle()
            particle.setptetaphim(met, 0.0, phi, 0.0)
            particle.set_pdgid(0)
            event_particles.append(particle)

        all_particles.append(event_particles)

    return all_particles


def _get_particles_met(tree):
    mets = tree.array("MissingET.MET")
    phis = tree.array("MissingET.Phi")

    all_particles = []

    for ievent in range(len(mets)):
        event_particles = []

        for met, phi in zip(mets[ievent], phis[ievent]):
            particle = MadMinerParticle()
            particle.setptetaphim(met, 0.0, phi, 0.0)
            particle.set_pdgid(0)
            event_particles.append(particle)

        all_particles.append(event_particles)

    return all_particles
