from __future__ import annotations

import os
import shutil
import logging
from collections import defaultdict
from pathlib import Path
from typing import Union

import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdchem import KekulizeException

from . import io
from .polymer_core import prepare_monomer_nocap, prepare_cap_monomer

BENZENE_PATTERN = Chem.MolFromSmarts("c1ccccc1")
logger = logging.getLogger(__name__)


def _forcefield_resource_root() -> Path:
    local_root = Path(__file__).resolve().parent / "resources" / "forcefields"
    if local_root.exists():
        return local_root
    raise FileNotFoundError(f"Could not locate forcefield resource directory: {local_root}")


def gen_ff_from_data(work_dir: Union[str, os.PathLike], compound_name: str, corr_factor: float, target_sum_chg: float):
    work_path = Path(work_dir)
    md_dir = work_path / "MD_dir"
    md_dir.mkdir(parents=True, exist_ok=True)

    resource_root = _forcefield_resource_root()
    files_to_copy = [
        resource_root / "pdb" / f"{compound_name}.pdb",
        resource_root / "itp" / f"{compound_name}_bonded.itp",
        resource_root / "itp" / f"{compound_name}_nonbonded.itp",
    ]
    for src in files_to_copy:
        if not src.exists():
            raise FileNotFoundError(f"Missing forcefield resource: {src}")
        shutil.copy(src, md_dir)

    scale_chg_itp(md_dir, f"{compound_name}_bonded.itp", corr_factor, target_sum_chg)


def find_substruct_matches(
    target_mol,
    query_mol,
    *,
    uniquify=True,
    ignore_stereo=True,
    allow_aromatic_conj=True,
    match_hs=True,
    try_remove_stereo=True,
):
    if target_mol is None or query_mol is None:
        return []

    auto_uniquify = True
    if target_mol is not None and BENZENE_PATTERN is not None:
        auto_uniquify = not target_mol.HasSubstructMatch(BENZENE_PATTERN)
    uniquify = auto_uniquify

    def _get_matches(t, q, params):
        try:
            ms = t.GetSubstructMatches(q, params=params)
        except TypeError:
            ms = t.GetSubstructMatches(
                q,
                useChirality=getattr(params, "useChirality", False),
                uniquify=uniquify,
            )

        if not ms:
            return []

        ms = list(ms)
        if len(ms) > 1:
            ms.sort(key=lambda m: (len(m), m))
        return ms

    p = Chem.SubstructMatchParameters()
    p.uniquify = uniquify

    if hasattr(p, "useChirality"):
        p.useChirality = not ignore_stereo
    if hasattr(p, "useEnhancedStereo"):
        p.useEnhancedStereo = not ignore_stereo
    if hasattr(p, "aromaticMatchesConjugated"):
        p.aromaticMatchesConjugated = allow_aromatic_conj
    if hasattr(p, "useHs"):
        p.useHs = match_hs
    if hasattr(p, "maxMatches"):
        p.maxMatches = 0

    matches = _get_matches(target_mol, query_mol, p)
    if matches:
        return matches

    if try_remove_stereo:
        t2 = Chem.Mol(target_mol)
        q2 = Chem.Mol(query_mol)
        Chem.RemoveStereochemistry(t2)
        Chem.RemoveStereochemistry(q2)

        matches = _get_matches(t2, q2, p)
        del t2, q2
        if matches:
            return matches

    t3 = Chem.Mol(target_mol)
    q3 = Chem.Mol(query_mol)
    try:
        Chem.Kekulize(t3, clearAromaticFlags=True)
        Chem.Kekulize(q3, clearAromaticFlags=True)
    except KekulizeException:
        del t3, q3
        return []

    if hasattr(p, "aromaticMatchesConjugated"):
        p.aromaticMatchesConjugated = False

    matches = _get_matches(t3, q3, p)
    del t3, q3
    return matches


def select_non_overlapping_matches(matches, used_atoms=None):
    if not matches:
        return []

    used_atoms = set() if used_atoms is None else set(used_atoms)

    selected = []
    for match in sorted(matches, key=lambda m: (min(m), m)):
        if any(atom_idx in used_atoms for atom_idx in match):
            continue
        selected.append(match)
        used_atoms.update(match)
    return selected


def assign_partial_charges(mol_poly, sub_mol, matches):
    if sub_mol is None:
        return
    for match in matches:
        for sub_atom_idx, poly_atom_idx in enumerate(match):
            sub_atom = sub_mol.GetAtomWithIdx(sub_atom_idx)
            if not sub_atom.HasProp("partial_charge"):
                continue
            charge = float(sub_atom.GetProp("partial_charge"))
            mol_poly.GetAtomWithIdx(poly_atom_idx).SetDoubleProp("partial_charge", charge)


def mol_to_charge_df(mol):
    data = []
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_symbol = atom.GetSymbol()
        charge = float(atom.GetProp("partial_charge")) if atom.HasProp("partial_charge") else None
        data.append({"atom_index": atom_idx, "atom": atom_symbol, "charge": charge})

    df = pd.DataFrame(data)
    df = df.sort_values("atom_index").reset_index(drop=True)
    return df


def _connected_components_for_indices(mol, atom_indices):
    remaining = set(atom_indices)
    components = []

    while remaining:
        start = remaining.pop()
        stack = [start]
        component = [start]

        while stack:
            atom_idx = stack.pop()
            atom = mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx in remaining:
                    remaining.remove(neighbor_idx)
                    stack.append(neighbor_idx)
                    component.append(neighbor_idx)

        components.append(sorted(component))

    components.sort(key=lambda comp: (min(comp), len(comp), comp))
    return components


def _split_terminal_components(mol, best_matches, no_matched_atoms):
    if not no_matched_atoms:
        return [], []

    if len(best_matches) != 1:
        left_neighbor = set(best_matches[0])
        right_neighbor = set(best_matches[-1])
        left_end_atoms = []
        right_end_atoms = []

        for idx in list(no_matched_atoms):
            atom = mol.GetAtomWithIdx(idx)
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx in left_neighbor:
                    left_end_atoms.append(idx)
                    left_neighbor.add(idx)
                    break
                if neighbor_idx in right_neighbor:
                    right_end_atoms.append(idx)
                    right_neighbor.add(idx)
                    break

        return sorted(left_end_atoms), sorted(right_end_atoms)

    components = _connected_components_for_indices(mol, no_matched_atoms)
    if len(components) == 1:
        logger.warning(
            "Single-match repeat template left only one unmatched component; assigning it to both terminal templates."
        )
        return list(components[0]), list(components[0])

    if len(components) > 2:
        logger.warning(
            "Single-match repeat template produced %s unmatched components; using min/max index components as terminal templates.",
            len(components),
        )

    left_component = min(components, key=lambda comp: min(comp))
    right_component = max(components, key=lambda comp: max(comp))
    return list(left_component), list(right_component)


def _repeat_connection_info(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid repeating unit SMILES: %s" % smiles)
    dummy_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummy_indices) != 2:
        raise ValueError("Repeating unit must contain exactly two '*' atoms.")
    atom1 = mol.GetAtomWithIdx(dummy_indices[0]).GetNeighbors()[0].GetIdx()
    atom2 = mol.GetAtomWithIdx(dummy_indices[1]).GetNeighbors()[0].GetIdx()
    return dummy_indices[0], dummy_indices[1], atom1, atom2


def _segment_short_chain_by_builder_order(
    mol_poly,
    repeating_unit,
    length_short,
    end_repeating,
    left_cap_smiles=None,
    right_cap_smiles=None,
):
    dum1, dum2, atom1, atom2 = _repeat_connection_info(repeating_unit)
    repeat_mol, _, _ = prepare_monomer_nocap(repeating_unit, dum1, dum2, atom1, atom2)
    repeat_atom_count = repeat_mol.GetNumAtoms()

    left_cap_count = 0
    right_cap_count = 0
    if left_cap_smiles:
        left_cap_count = prepare_cap_monomer(left_cap_smiles)[0].GetNumAtoms()
    if right_cap_smiles:
        right_cap_count = prepare_cap_monomer(right_cap_smiles)[0].GetNumAtoms()

    expected_atoms = repeat_atom_count * length_short + left_cap_count + right_cap_count
    if mol_poly.GetNumAtoms() != expected_atoms:
        logger.warning(
            "Builder-order segmentation skipped: expected_atoms=%s actual_atoms=%s repeat_atom_count=%s length_short=%s left_cap=%s right_cap=%s",
            expected_atoms,
            mol_poly.GetNumAtoms(),
            repeat_atom_count,
            length_short,
            left_cap_count,
            right_cap_count,
        )
        return None

    repeat_blocks = []
    for block_idx in range(length_short):
        start = block_idx * repeat_atom_count
        stop = start + repeat_atom_count
        repeat_blocks.append(list(range(start, stop)))

    chain_end = repeat_atom_count * length_short
    left_cap_atoms = list(range(chain_end, chain_end + left_cap_count))
    right_cap_atoms = list(range(chain_end + left_cap_count, chain_end + left_cap_count + right_cap_count))

    left_atoms = left_cap_atoms + [atom_idx for block in repeat_blocks[:end_repeating] for atom_idx in block]
    right_atoms = [atom_idx for block in repeat_blocks[-end_repeating:] for atom_idx in block] + right_cap_atoms

    if length_short >= 3:
        mid_template_atoms = list(repeat_blocks[1])
    else:
        mid_template_atoms = list(repeat_blocks[0])

    logger.info(
        "Builder-order segmentation ready: repeat_atom_count=%s repeat_blocks=%s left_cap=%s right_cap=%s left_atoms=%s right_atoms=%s mid_atoms=%s",
        repeat_atom_count,
        len(repeat_blocks),
        len(left_cap_atoms),
        len(right_cap_atoms),
        len(left_atoms),
        len(right_atoms),
        len(mid_template_atoms),
    )
    return left_atoms, right_atoms, mid_template_atoms


def apply_chg_to_poly(
    work_dir,
    mol_short,
    mol_long,
    itp_file,
    resp_chg_df,
    repeating_unit,
    end_repeating,
    scale,
    charge,
    length_short=None,
    left_cap_smiles=None,
    right_cap_smiles=None,
):
    logger.info(
        "Applying polymer charge transfer: work_dir=%s itp=%s repeating_unit=%s end_repeating=%s scale=%s target_charge=%s",
        work_dir,
        itp_file,
        repeating_unit,
        end_repeating,
        scale,
        charge,
    )
    md_dir = os.path.join(work_dir, "MD_dir")
    os.makedirs(md_dir, exist_ok=True)

    left_mol, right_mol, mid_mol = apply_chg2mol(
        resp_chg_df,
        mol_short,
        repeating_unit,
        end_repeating,
        length_short=length_short,
        left_cap_smiles=left_cap_smiles,
        right_cap_smiles=right_cap_smiles,
    )
    logger.info(
        "Charge transfer fragments ready: left=%s right=%s mid=%s",
        left_mol.GetNumAtoms() if left_mol is not None else None,
        right_mol.GetNumAtoms() if right_mol is not None else None,
        mid_mol.GetNumAtoms() if mid_mol is not None else None,
    )

    Chem.SanitizeMol(mol_long)
    mol_poly = Chem.AddHs(mol_long)

    used_atoms = set()

    left_matches = []
    all_left = find_substruct_matches(mol_poly, left_mol)
    if all_left:
        left_match = min(all_left, key=lambda m: sum(m) / len(m))
        left_matches.append(left_match)
        used_atoms.update(left_match)

    right_matches = []
    all_right = find_substruct_matches(mol_poly, right_mol)
    if all_right:
        right_match = max(all_right, key=lambda m: sum(m) / len(m))
        if not any(atom_idx in used_atoms for atom_idx in right_match):
            right_matches.append(right_match)
            used_atoms.update(right_match)

    mid_matches = []
    raw_mid_matches = find_substruct_matches(mol_poly, mid_mol)
    for match in select_non_overlapping_matches(raw_mid_matches, used_atoms):
        mid_matches.append(match)
        used_atoms.update(match)

    assign_partial_charges(mol_poly, left_mol, left_matches)
    assign_partial_charges(mol_poly, right_mol, right_matches)
    assign_partial_charges(mol_poly, mid_mol, mid_matches)

    charge_update_df = mol_to_charge_df(mol_poly)
    logger.info(
        "Charge dataframe before neutralization: atoms=%s nan_count=%s sum=%s",
        len(charge_update_df),
        int(charge_update_df["charge"].isna().sum()),
        charge_update_df["charge"].sum(skipna=True),
    )
    charge_update_df_cor = charge_neutralize_scale(charge_update_df, scale, charge)
    logger.info(
        "Charge dataframe after neutralization: atoms=%s nan_count=%s sum=%s",
        len(charge_update_df_cor),
        int(charge_update_df_cor["charge"].isna().sum()),
        charge_update_df_cor["charge"].sum(skipna=True),
    )
    update_itp_file(md_dir, itp_file, charge_update_df_cor)


def apply_chg2mol(
    resp_chg_df,
    mol_poly,
    repeating_unit,
    end_repeating,
    length_short=None,
    left_cap_smiles=None,
    right_cap_smiles=None,
):
    resp_chg_df = resp_chg_df.copy()
    max_idx = resp_chg_df["position"].max()
    if max_idx == mol_poly.GetNumAtoms():
        resp_chg_df["position"] = resp_chg_df["position"] - 1

    for _, row in resp_chg_df.iterrows():
        pos = int(row["position"])
        charge = float(row["charge"])
        if pos < 0 or pos >= mol_poly.GetNumAtoms():
            continue
        atom = mol_poly.GetAtomWithIdx(pos)
        atom.SetDoubleProp("partial_charge", charge)

    partial_charges = [
        float(row["charge"])
        for _, row in resp_chg_df.sort_values("position").iterrows()
    ]
    mol_poly.SetProp("partial_charges", ",".join(map(str, partial_charges)))

    segmented = None
    if length_short is not None:
        segmented = _segment_short_chain_by_builder_order(
            mol_poly,
            repeating_unit,
            int(length_short),
            int(end_repeating),
            left_cap_smiles=left_cap_smiles,
            right_cap_smiles=right_cap_smiles,
        )

    if segmented is not None:
        left_atoms, right_atoms, template_atoms = segmented
        left_mol = gen_molfromindex(mol_poly, left_atoms)
        right_mol = gen_molfromindex(mol_poly, right_atoms)
        mid_mol = gen_molfromindex(mol_poly, template_atoms)

        for i, atom_idx in enumerate(left_atoms):
            charge = mol_poly.GetAtomWithIdx(atom_idx).GetDoubleProp("partial_charge")
            if i < left_mol.GetNumAtoms():
                left_mol.GetAtomWithIdx(i).SetDoubleProp("partial_charge", charge)

        for i, atom_idx in enumerate(right_atoms):
            charge = mol_poly.GetAtomWithIdx(atom_idx).GetDoubleProp("partial_charge")
            if i < right_mol.GetNumAtoms():
                right_mol.GetAtomWithIdx(i).SetDoubleProp("partial_charge", charge)

        for i, atom_idx in enumerate(template_atoms):
            charge = mol_poly.GetAtomWithIdx(atom_idx).GetDoubleProp("partial_charge")
            if i < mid_mol.GetNumAtoms():
                mid_mol.GetAtomWithIdx(i).SetDoubleProp("partial_charge", charge)

        logger.info(
            "Charge-transfer templates built from builder order: left=%s right=%s mid=%s",
            left_mol.GetNumAtoms(),
            right_mol.GetNumAtoms(),
            mid_mol.GetNumAtoms(),
        )
        return left_mol, right_mol, mid_mol

    mol_unit_fwd = Chem.MolFromSmiles(repeating_unit)
    mol_unit_fwd = Chem.AddHs(mol_unit_fwd)
    edit_fwd = Chem.EditableMol(mol_unit_fwd)
    for atom in reversed(list(mol_unit_fwd.GetAtoms())):
        if atom.GetSymbol() == "*":
            edit_fwd.RemoveAtom(atom.GetIdx())
    mol_unit_fwd = edit_fwd.GetMol()

    num_atoms_fwd = mol_unit_fwd.GetNumAtoms()
    new_order = list(range(num_atoms_fwd - 1, -1, -1))
    mol_unit_rev = Chem.RenumberAtoms(mol_unit_fwd, new_order)

    rw_mol = Chem.RWMol(mol_poly)

    fwd_used_atoms = set()
    fwd_matches = []
    for match in rw_mol.GetSubstructMatches(mol_unit_fwd, uniquify=True, useChirality=False):
        if any(atom_idx in fwd_used_atoms for atom_idx in match):
            continue
        fwd_matches.append(match)
        fwd_used_atoms.update(match)

    rev_used_atoms = set()
    rev_matches = []
    for match in rw_mol.GetSubstructMatches(mol_unit_rev, uniquify=True, useChirality=False):
        if any(atom_idx in rev_used_atoms for atom_idx in match):
            continue
        rev_matches.append(match)
        rev_used_atoms.update(match)

    if len(fwd_matches) >= len(rev_matches):
        best_matches = fwd_matches
        best_label = "forward"
    else:
        best_matches = rev_matches
        best_label = "reverse"

    logger.info(
        "Repeating-unit matches: fwd=%s rev=%s best=%s count=%s",
        len(fwd_matches),
        len(rev_matches),
        best_label,
        len(best_matches),
    )

    if not best_matches:
        logger.error("No repeating-unit matches found for charge transfer.")
        return None, None, None

    matched_atoms = set()
    for match in best_matches:
        matched_atoms.update(match)

    no_matched_atoms = [
        atom.GetIdx() for atom in mol_poly.GetAtoms() if atom.GetIdx() not in matched_atoms
    ]

    left_end_atoms, right_end_atoms = _split_terminal_components(
        mol_poly,
        best_matches,
        no_matched_atoms,
    )
    logger.info(
        "Charge-transfer terminal split: unmatched=%s left_end=%s right_end=%s",
        len(no_matched_atoms),
        len(left_end_atoms),
        len(right_end_atoms),
    )

    left_atoms = left_end_atoms + [atom_idx for match in best_matches[:end_repeating] for atom_idx in match]
    right_atoms = right_end_atoms + [atom_idx for match in best_matches[-end_repeating:] for atom_idx in match]

    left_mol = gen_molfromindex(mol_poly, left_atoms)
    right_mol = gen_molfromindex(mol_poly, right_atoms)

    for i, atom_idx in enumerate(left_atoms):
        charge = mol_poly.GetAtomWithIdx(atom_idx).GetDoubleProp("partial_charge")
        if i < left_mol.GetNumAtoms():
            left_mol.GetAtomWithIdx(i).SetDoubleProp("partial_charge", charge)

    for i, atom_idx in enumerate(right_atoms):
        charge = mol_poly.GetAtomWithIdx(atom_idx).GetDoubleProp("partial_charge")
        if i < right_mol.GetNumAtoms():
            right_mol.GetAtomWithIdx(i).SetDoubleProp("partial_charge", charge)

    mid_atoms = best_matches[1:-1]
    if len(mid_atoms) == 0 and len(best_matches) == 2:
        mid_atoms = [best_matches[0], best_matches[1]]
        logger.warning("Charge-transfer mid fallback activated: using both edge matches as mid (count=2)")
    elif len(mid_atoms) == 0 and len(best_matches) == 1:
        mid_atoms = [best_matches[0]]
        logger.warning("Charge-transfer mid fallback activated: using single best match as mid (count=1)")
    num_repeats = len(mid_atoms)
    logger.info(
        "Charge-transfer region summary: left_atoms=%s right_atoms=%s mid_repeat_matches=%s",
        len(left_atoms),
        len(right_atoms),
        num_repeats,
    )

    charge_dict = defaultdict(list)
    for match in mid_atoms:
        for pos, atom_idx in enumerate(match):
            charge = mol_poly.GetAtomWithIdx(atom_idx).GetDoubleProp("partial_charge")
            charge_dict[pos].append(charge)

    avg_charges = {pos: sum(charges) / len(charges) for pos, charges in charge_dict.items()}

    if num_repeats > 0:
        template_atoms = list(mid_atoms[0])
        mid_mol = gen_molfromindex(mol_poly, template_atoms)
        for pos, atom in enumerate(mid_mol.GetAtoms()):
            atom.SetDoubleProp("partial_charge", avg_charges[pos])
    else:
        mid_mol = None
        logger.error("No mid repeating-unit charges could be constructed.")

    return left_mol, right_mol, mid_mol


def update_itp_file(md_dir, itp_file, charge_update_df_cor):
    itp_filepath = os.path.join(md_dir, itp_file)
    with open(itp_filepath, "r") as file:
        lines = file.readlines()

    in_section = False
    start_index = end_index = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("[") and "atoms" in line.split():
            in_section = True
            continue
        if in_section:
            if line.strip().startswith(";"):
                start_index = i + 1
                continue
            if line.strip() == "":
                end_index = i
                break

    charge_index = 0
    nan_count = int(charge_update_df_cor["charge"].isna().sum())
    logger.info("Updating ITP charges: file=%s rows=%s nan_count=%s", itp_filepath, len(charge_update_df_cor), nan_count)
    for i in range(start_index, end_index):
        parts = lines[i].split()
        if charge_index < len(charge_update_df_cor):
            new_charge = charge_update_df_cor.iloc[charge_index]["charge"]
            parts[6] = f"{new_charge:.8f}"
            lines[i] = " ".join(parts) + "\n"
            charge_index += 1

    with open(itp_filepath, "w") as file:
        file.writelines(lines)


def gen_molfromindex(mol, idx):
    editable_mol = Chem.EditableMol(Chem.Mol())

    atom_map = {}
    for old_idx in idx:
        atom = mol.GetAtomWithIdx(old_idx)
        new_idx = editable_mol.AddAtom(atom)
        atom_map[old_idx] = new_idx

    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if begin_idx in idx and end_idx in idx:
            new_begin = atom_map[begin_idx]
            new_end = atom_map[end_idx]
            editable_mol.AddBond(new_begin, new_end, bond.GetBondType())

    return editable_mol.GetMol()


def charge_neutralize_scale(df, correction_factor=1, target_total_charge=0):
    current_total_charge = df["charge"].sum()
    charge_difference = target_total_charge - current_total_charge
    charge_adjustment_per_atom = charge_difference / len(df)
    logger.info(
        "Neutralizing charges: current_sum=%s target_sum=%s correction_factor=%s per_atom_adjustment=%s",
        current_total_charge,
        target_total_charge,
        correction_factor,
        charge_adjustment_per_atom,
    )
    df["charge"] = (df["charge"] + charge_adjustment_per_atom) * correction_factor
    return df


def scale_chg_itp(work_dir, filename, corr_factor, target_sum_chg):
    filename = os.path.join(work_dir, filename)
    start_reading = False
    atoms = []

    with open(filename, "r") as file:
        for line in file:
            if line.strip().startswith("[") and "atoms" in line.split():
                start_reading = True
                continue
            if start_reading:
                if line.strip() == "":
                    break
                if line.strip().startswith(";"):
                    continue
                parts = line.split()
                if len(parts) >= 7:
                    atom_id = parts[4]
                    charge = float(parts[6])
                    atoms.append([atom_id, charge])

    df = pd.DataFrame(atoms, columns=["atom", "charge"])
    charge_update_df_cor = charge_neutralize_scale(df, corr_factor, target_sum_chg)

    with open(filename, "r") as file:
        lines = file.readlines()

    in_section = False
    start_index = end_index = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("[") and "atoms" in line.split():
            in_section = True
            continue
        if in_section:
            if line.strip().startswith(";"):
                start_index = i + 1
                continue
            if line.strip() == "":
                end_index = i
                break

    charge_index = 0
    for i in range(start_index, end_index):
        parts = lines[i].split()
        if charge_index < len(charge_update_df_cor):
            new_charge = charge_update_df_cor.iloc[charge_index]["charge"]
            parts[6] = f"{new_charge:.8f}"
            lines[i] = " ".join(parts) + "\n"
            charge_index += 1

    with open(filename, "w") as file:
        file.writelines(lines)
