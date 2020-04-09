from __future__ import absolute_import, division, print_function, unicode_literals
import six

import numpy as np
from collections import OrderedDict
import uproot
import os
import logging
import itertools


from madminer.utils.particle import MadMinerParticle
from madminer.utils.various import math_commands

logger = logging.getLogger(__name__)
import itertools


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

        ##################### PREPARE FUNCTIONS TO FIND BEST ZZ CANDIDATE #####################

        #find same flavour opposite sign lepton pairs,
        #given the list of electrons or muons in an event
        #returns all possible SFOS pairs ([pos_idx, neg_idx],...)
        #which satisfy invariant mass cuts
        def find_Z_cand(cand_list, part_list):
            #positively and negatively charged leptons, and their
            idx_pos = []
            cand_pos = []
            idx_neg = []
            cand_neg = []
            for idx, (is_cand, part) in enumerate(zip(cand_list, part_list)):
                #checking if the lepton is a candidate
                if part.charge > 0:
                    idx_pos.append(idx)
                    cand_pos.append(is_cand)
                if part.charge < 0:
                    idx_neg.append(idx)
                    cand_neg.append(is_cand)

            z_cand_list = []
            for candpair, pair in zip(itertools.product(cand_pos, cand_neg), itertools.product(idx_pos, idx_neg)):
                iscand = candpair[0]*candpair[1] #1 only if both are candidates
                #cut invariant mass
                invm = (part_list[pair[0]] + part_list[pair[1]]).m
                if invm < 120 and invm > 12:
                    z_cand_list.append([pair[0],pair[1]])

            return z_cand_list #[[pos_idx,neg_idx], [pos_idx,neg_idx], ...]

        #find pairs of SFOS pairs with no leptons in common
        #keep track of their different flavours
        #order of Z1 and Z2 is not important (each only counted once)
        def find_ZZ_cand(Z_cand_e_list, Z_cand_mu_list):
            ZZ_cand_ee = []; ZZ_cand_mm = []; ZZ_cand_em = []
            #same flavour e
            for Zcand1,Zcand2 in itertools.product(Z_cand_e_list,Z_cand_e_list):
                if not any(x in Zcand1 for x in Zcand2):
                    ZZ_cand_ee.append([Zcand1[0], Zcand1[1], Zcand2[0], Zcand2[1]])

            #same flavour mu
            for Zcand1,Zcand2 in itertools.product(Z_cand_mu_list,Z_cand_mu_list):
                if not any(x in Zcand1 for x in Zcand2):
                    ZZ_cand_mm.append([Zcand1[0], Zcand1[1], Zcand2[0], Zcand2[1]])

            #different flavour
            for Zcand1,Zcand2 in itertools.product(Z_cand_e_list,Z_cand_mu_list):
                ZZ_cand_em.append([Zcand1[0], Zcand1[1], Zcand2[0], Zcand2[1]])
            return ZZ_cand_ee, ZZ_cand_mm, ZZ_cand_em

        #return the list of particles in the correct order:
        #first list with flavour of first Z, second flavour of second Z
        def set_flavours_lists(part_list_e, part_list_mu, flavours):
            if flavours == "ee":
                part_list1 = part_list_e
                part_list2 = part_list_e
            elif flavours == "mm":
                part_list1 = part_list_mu
                part_list2 = part_list_mu
            elif flavours == "em":
                part_list1 = part_list_e
                part_list2 = part_list_mu
            elif flavours == "me":
                part_list1 = part_list_mu
                part_list2 = part_list_e
            else:
                print("set_flavours_lists: flavour not valid: choose between ee, mm, em")
                return
            return part_list1, part_list2

        #return the leptons corresponding to the index in ZZcand
        #taking into account the flavours
        def ZZidx_to_leps(ZZcand, part_list_e, part_list_mu, flavours):
            part_list1, part_list2 = set_flavours_lists(part_list_e, part_list_mu, flavours)
            leps = [part_list1[ZZcand[0]], part_list1[ZZcand[1]], part_list2[ZZcand[2]], part_list2[ZZcand[3]]]
            return leps

        #reorder the two Z in each pair: first the one with closest mass to Z
        #keep track of the possible swapping in case of different flavours
        def Z1Z2_ordering(ZZcand, part_list_e, part_list_mu, flavours):
            leps = ZZidx_to_leps(ZZcand, part_list_e, part_list_mu, flavours)
            #compute invariant masses of each Z
            mz_p1 = (leps[0] + leps[1]).m
            mz_p2 = (leps[2] + leps[3]).m
            #order accordingly to closest mass to Z mass
            if min([mz_p1,mz_p2], key=lambda x:abs(x-91.2), default=-1) == mz_p1:
                z1z2 = ZZcand
                swapped = False
            else:
                z1z2 = [ZZcand[2],ZZcand[3],ZZcand[0],ZZcand[1]]
                swapped = True
            return(z1z2, swapped)

        #apply cuts to ZZ candidate
        #consider differently the case of same flavour ZZ
        def ZZ_cuts(ZZcand, part_list_e, part_list_mu, flavours, isSF):
            leps = ZZidx_to_leps(ZZcand, part_list_e, part_list_mu, flavours)
            #Z1 mass
            mz1 = (leps[0] + leps[1]).m
            if not mz1 > 40:
                #print("fail mz1 mass cut")
                return False
            #leptons pt
            leps_pt_sort = sorted(leps, key=lambda x:x.pt)
            if not (leps_pt_sort[0].pt > 20 and leps_pt_sort[1].pt > 10):
                #print("fail leptons pt cut")
                return False
            #OS invariant mass (pos, neg, pos, neg)
            mza = (leps[0]+leps[3]).m
            mzb = (leps[1]+leps[2]).m
            mz2 = (leps[2]+leps[3]).m
            if not (mza > 4 and mzb > 4 and mz2 > 4):
                #print("fail os cut")
                return False
            #4l invariant mass
            m4l = (leps[0]+leps[1]+leps[2]+leps[3]).m
            #print("m4l: ", m4l)
            if not m4l>70:
                #print("fail m4l cut")
                return False
            #same flavour cut
            if isSF == True:
                mz_pdg = 91.2
                mzab_ord = [mza, mzb] if min([mza,mzb], key=lambda x:abs(x-mz_pdg)) == mza else [mzb,mza]
                if (abs(mzab_ord[0]-mz_pdg) < abs(mz1-mz_pdg) and mzb < mzab_ord[0]):
                    #print("fail SF cut")
                    return False
            #if candidate passes all cuts
            return True

        #choose ZZ candidate for which Z1 has mass closest to Z
        #keep track of the flavour of Z1 and Z2
        def choose_final_ZZ(ZZlist, part_list_e, part_list_mu, flavourslist):
            if len(ZZlist) == 1:
                return ZZlist[0], flavourslist[0]
            else:
                mz1list = []
                for ZZcand, flavours in zip(ZZlist,flavourslist):
                    part_list1, part_list2 = set_flavours_lists(part_list_e, part_list_mu, flavours)
                    mz1 = (part_list1[ZZcand[0]] + part_list1[ZZcand[1]]).m
                    mz1list.append(mz1)
                mz1min = min(mz1list, key=lambda x:abs(x-91.2))
                idx = mz1list.index(mz1min)
                return ZZlist[idx], flavourslist[idx]

        ##################### MAIN CODE TO FIND BEST ZZ CANDIDATE #####################
        #default values
        isZZcand = 0
        lep1 = 0; lep2 = 0; lep3 = 0; lep4 = 0

        #list of candidates electrons/muons: 1 if candidate, 0 if not
        candidate_es = np.ones(len(objects["e"]))
        candidate_mus = np.ones(len(objects["mu"]))
        #print(len(objects["e"]), len(objects["mu"]))

        #find candidate leptons: eta and pt cuts (to be checked)
        for idx, el in enumerate(objects["e"]):
            if not (abs(el.eta) < 2.5 and el.pt > 7):
                candidate_es[idx] = 0
        for idx, mu in enumerate(objects["mu"]):
            if not (abs(mu.eta) < 2.4 and mu.pt > 5):
                candidate_mus[idx] = 0

        #find Z candidates
        e_Z_cand = find_Z_cand(candidate_es, objects["e"])
        mu_Z_cand = find_Z_cand(candidate_mus, objects["mu"])

        #find ZZ candidates
        ZZ_cand_ee_list, ZZ_cand_mm_list, ZZ_cand_em_list = find_ZZ_cand(e_Z_cand,mu_Z_cand)
 
        #initialise lists for ZZ candidates which pass cuts, and keep track of flavours
        ZZ_cands_final = []
        ZZ_cands_final_flavours = []

        #same flavours, electrons
        for ZZ_cand_ee in ZZ_cand_ee_list:
            ZZ_cand_ee, swapped = Z1Z2_ordering(ZZ_cand_ee, objects["e"], objects["mu"], flavours="ee")
            #apply cuts
            if ZZ_cuts(ZZ_cand_ee, objects["e"], objects["mu"], flavours="ee", isSF=True) == True:
                ZZ_cands_final.append(ZZ_cand_ee)
                ZZ_cands_final_flavours.append("ee")
        #same flavours, muons
        for ZZ_cand_mm in ZZ_cand_mm_list:
            ZZ_cand_mm, swapped = Z1Z2_ordering(ZZ_cand_mm, objects["e"], objects["mu"], flavours="mm")
            #apply cuts
            if ZZ_cuts(ZZ_cand_mm, objects["e"], objects["mu"], flavours="mm", isSF=True) == True:
                ZZ_cands_final.append(ZZ_cand_mm)
                ZZ_cands_final_flavours.append("mm")
        #different flavours
        for ZZ_cand_em in ZZ_cand_em_list:
            ZZ_cand_em, swapped = Z1Z2_ordering(ZZ_cand_em, objects["e"], objects["mu"], flavours="em")
            #apply cuts, careful when swapping em/me
            flavours_OF = "em" if swapped==False else "me"
            if ZZ_cuts(ZZ_cand_em, objects["e"], objects["mu"], flavours=flavours_OF, isSF=False) == True:
                ZZ_cands_final.append(ZZ_cand_em)
                ZZ_cands_final_flavours.append(flavours_OF)

        #find final best ZZ candidate
        #if more than one ZZ candidate is left: choose the one with Z1 mass closest to MZ
        if len(ZZ_cands_final) > 0:
            isZZcand = 1
            ZZfinal, flavours = choose_final_ZZ(ZZ_cands_final, objects["e"], objects["mu"], ZZ_cands_final_flavours)
            lep1, lep2, lep3, lep4 = ZZidx_to_leps(ZZfinal, objects["e"], objects["mu"], flavours)


        #logger.debug("Value of isZZcand: %d", isZZcand)
        #debugging
        #if(len(ZZ_cand_ee_list) + len(ZZ_cand_mm_list) + len(ZZ_cand_em_list)) >= 1:
            #if isZZcand != 1:
                #print("at least 2 and 2 but no ZZ cand")
            #else:
                #print("ok candidate and found")

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

        logger.debug("  First 10 values for observable %s:\n%s", obs_name, values_this_observable[:10])

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
