import collections
import json
import pickle

import pandas as pd
import pytest

from molforge.actors.confs import GenerateConfs
from molforge.actors.params.confs import GenerateConfsParams
from molforge.actors.protocol import ActorInput
from molforge.configuration.context import PipelineContext
from molforge.configuration.logger import PipelineLogger

try:
    from rdkit import Chem
    from phd_tools.chemistry.flexibility import FlexibilityCalculator
    from molforge.actor_plugins.flexibility import (ENDPOINTS, FORMAT_VERSION,
                                                    CalculateFlexibility,
                                                    CalculateFlexibilityParams)
    FLEX_AVAILABLE = True
except ImportError:
    FLEX_AVAILABLE = False


@pytest.fixture
def molecules():
    return pd.DataFrame({
        'smiles': [
            'CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O',
            'CC(=O)Nc1ccc(O)cc1',
            'OC(=O)Cc1ccccc1',
            'CC1CCN(CC1)C(=O)OCC',
        ],
        'molecule_id': ['s_ibuprofen', 'paracetamol', 'phenylacetic_acid', 'piperidine_carbamate'],
    })


@pytest.fixture
def pipeline(tmp_path):
    out_dir = tmp_path / "test_flexibility"
    out_dir.mkdir()
    logger = PipelineLogger(name='test_flexibility', log_file=str(out_dir / 'test.log'),
                            console_level='INFO')
    return logger, PipelineContext(run_id='test_flexibility', output_dir=str(out_dir))


def signature(mol, torsion):
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False, includeChirality=False))
    return tuple(ranks[atom] for atom in torsion)


def read_endpoint(actor, endpoint):
    directory = actor.endpoint_dir(endpoint)
    out = {}
    for entry in json.load(open(directory / "manifest.json")):
        assert entry["format_version"] == FORMAT_VERSION
        chunk = pickle.load(open(directory / entry["file"], "rb"))
        assert chunk["format_version"] == FORMAT_VERSION and chunk["names"] == entry["names"]
        assert len(chunk["molecules"]) == len(chunk["values"]) == len(chunk["torsion_atoms"]) \
            == entry["count"]
        for k, name in enumerate(chunk["names"]):
            out[name] = (chunk["molecules"][k], chunk["values"][k], chunk["torsion_atoms"][k])
    return out


@pytest.mark.skipif(not FLEX_AVAILABLE, reason="Flexibility plugin (phd-tools) not available")
def test_sidecars_keep_each_label_on_its_bond_with_its_torsion(molecules, pipeline):
    logger, context = pipeline
    confs = GenerateConfs(GenerateConfsParams(backend='rdkit', max_confs=10, rms_threshold=0.5,
                                              SMILES_column='smiles', names_column='molecule_id',
                                              dropna=False), logger=logger)
    confs_output = confs(ActorInput(data=molecules, context=context))
    flexibility = CalculateFlexibility(CalculateFlexibilityParams(dropna=False), logger=logger)
    context.register_actor('confs', confs)
    context.register_actor('flexibility', flexibility)
    output = flexibility(ActorInput(data=confs_output.data, context=context))
    assert output.success

    rows = output.data.set_index('flexibility_name')
    stored = {e: read_endpoint(flexibility, e) for e in ENDPOINTS}
    assert set(stored["V"]) == set(rows.index[rows['flexibility_success']])
    assert all(set(stored[e]) == set(stored["V"]) for e in ENDPOINTS)

    calculator = FlexibilityCalculator(**{k: getattr(flexibility, k) for k in
                                          ('tau', 'n_confs_threshold', 'symmetry_radius',
                                           'ignore_colinear_bonds')})
    direct_labels = {name: calculator.compute(mol)
                     for name, mol in zip(confs.get_successful_names(), confs.extract_molecules())}

    for name, (mol, v_values, torsions) in stored["V"].items():
        s_mol, s_values, s_torsions = stored["S"][name]
        assert s_mol.ToBinary() == mol.ToBinary() and s_torsions == torsions
        assert set(v_values) == set(s_values) == set(torsions)
        for bond_idx, (a, b, c, d) in torsions.items():
            bond = mol.GetBondWithIdx(bond_idx)
            assert {b, c} == {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()}
            assert mol.GetBondBetweenAtoms(a, b) and mol.GetBondBetweenAtoms(c, d)

        labels, _, rd = direct_labels[name]
        kept = collections.Counter((signature(mol, t), v_values[b], s_values[b])
                                   for b, t in torsions.items())
        direct = collections.Counter((signature(rd, label['torsion_atoms']), label['V'], label['S'])
                                     for label in labels.values())
        assert kept == direct, name

        mapping = json.loads(rows.loc[name, 'flexibility_mapping'])
        assert all(len(p) == 4 for p in mapping['torsion_atom_positions'])
        assert len(mapping['V']) == len(mapping['S']) == len(torsions)

    provenance = json.load(open(flexibility.endpoint_dir("V").parent / "provenance.json"))
    assert provenance["format_version"] == FORMAT_VERSION
    assert provenance["endpoints"] == list(ENDPOINTS)
    assert set(provenance["code"]) == {"molforge", "phd_tools", "rdkit"}
