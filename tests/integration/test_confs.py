"""
Integration tests for unified conformer generation actor.

Tests both RDKit and OpenEye backends with the unified GenerateConfs actor.
"""

import pytest
import pandas as pd
from rdkit import Chem


from molforge.actors.confs import GenerateConfs
from molforge.actors.params.confs import GenerateConfsParams
from molforge.actors.protocol import ActorInput
from molforge.configuration.context import PipelineContext
from molforge.configuration.logger import PipelineLogger
from molforge.utils.constants import HAS_OPENEYE


@pytest.fixture
def test_data():
    """Sample SMILES data for testing."""
    return pd.DataFrame({
        'smiles': [
            'CCO',  # Ethanol - simple
            'c1ccccc1',  # Benzene - aromatic
            'CC(C)C',  # Isobutane - branched
            'INVALID',  # Invalid SMILES
        ],
        'molecule_id': ['mol_1', 'mol_2', 'mol_3', 'mol_4']
    })




@pytest.fixture
def output_dir(tmp_path):
    """Unified output directory for all test artifacts."""
    out_dir = tmp_path / "test_output"
    out_dir.mkdir(exist_ok=True)
    return out_dir

@pytest.fixture
def logger(output_dir):
    """Create logger for testing with unified output dir."""
    log_file = str(output_dir / 'test_confs_unified.log')
    return PipelineLogger(
        name='test_confs_unified',
        log_file=log_file,
        console_level='INFO'
    )




@pytest.fixture
def pipeline_context(output_dir):
    """Create pipeline context for testing with unified output dir."""
    return PipelineContext(
        run_id='test_confs_unified',
        output_dir=str(output_dir)
    )


class TestRDKitBackend:
    """Tests for RDKit backend via unified actor."""

    def test_rdkit_conformer_generation(self, test_data, logger, pipeline_context):
        """Test basic RDKit conformer generation."""
        params = GenerateConfsParams(
            backend='rdkit',
            max_confs=50,
            rms_threshold=0.5,
            SMILES_column='smiles',
            names_column='molecule_id',
            dropna=False
        )

        actor = GenerateConfs(params, logger=logger)
        actor_input = ActorInput(data=test_data, context=pipeline_context)

        output = actor(actor_input)

        # Check output structure
        assert output.success
        assert 'n_conformers' in output.data.columns
        assert 'conformer_success' in output.data.columns
        assert 'Status' in output.data.columns

        # Check metadata
        assert output.metadata['backend'] == 'rdkit'
        assert output.metadata['molecules_processed'] == 4
        assert output.metadata['success_count'] == 3  # 3 valid, 1 invalid

        # Check results
        df = output.data
        assert df.loc[df['molecule_id'] == 'mol_1', 'conformer_success'].iloc[0]
        assert df.loc[df['molecule_id'] == 'mol_2', 'conformer_success'].iloc[0]
        assert df.loc[df['molecule_id'] == 'mol_3', 'conformer_success'].iloc[0]
        assert not df.loc[df['molecule_id'] == 'mol_4', 'conformer_success'].iloc[0]

        # Check conformer counts
        assert df.loc[df['molecule_id'] == 'mol_1', 'n_conformers'].iloc[0] > 0
        assert df.loc[df['molecule_id'] == 'mol_4', 'n_conformers'].iloc[0] == 0

    def test_rdkit_extract_molecules(self, test_data, logger, pipeline_context):
        """Test molecule extraction from RDKit backend."""
        params = GenerateConfsParams(
            backend='rdkit',
            max_confs=20,
            SMILES_column='smiles',
            names_column='molecule_id',
            dropna=True
        )

        actor = GenerateConfs(params, logger=logger)
        actor_input = ActorInput(data=test_data, context=pipeline_context)

        output = actor(actor_input)

        # Extract molecules
        molecules = list(actor.extract_molecules())

        # Should have 3 successful molecules
        assert len(molecules) == 3

        # Check they are RDKit molecules with conformers
        for mol in molecules:
            assert isinstance(mol, Chem.Mol)
            assert mol.GetNumConformers() > 0
            assert mol.HasProp('_Name')

    def test_rdkit_dropna(self, test_data, logger, pipeline_context):
        """Test dropna parameter with RDKit backend."""
        params = GenerateConfsParams(
            backend='rdkit',
            max_confs=20,
            SMILES_column='smiles',
            names_column='molecule_id',
            dropna=True
        )

        actor = GenerateConfs(params, logger=logger)
        actor_input = ActorInput(data=test_data, context=pipeline_context)

        output = actor(actor_input)

        # Should only have successful molecules
        assert len(output.data) == 3
        assert output.data['conformer_success'].all()

    def test_rdkit_legacy_pattern(self, test_data, logger, pipeline_context):
        """Test that actor only accepts ActorInput (legacy removed)."""
        params = GenerateConfsParams(
            backend='rdkit',
            max_confs=20,
            SMILES_column='smiles',
            names_column='molecule_id',
            dropna=False
        )

        actor = GenerateConfs(params, logger=logger)

        # New pattern: must use ActorInput
        from molforge.actors.protocol import ActorInput, ActorOutput
        actor_input = ActorInput(data=test_data, context=pipeline_context)
        result = actor(actor_input)

        # Should return ActorOutput (not raw DataFrame)
        assert isinstance(result, ActorOutput)
        assert 'conformer_success' in result.data.columns
        assert 'n_conformers' in result.data.columns
        assert 'Status' in result.data.columns


class TestOpenEyeBackend:
    """Tests for OpenEye backend via unified actor."""

    @pytest.mark.skipif(
        not HAS_OPENEYE,
        reason="OpenEye toolkit not available"
    )
    def test_openeye_conformer_generation(self, test_data, logger, pipeline_context):
        """Test basic OpenEye conformer generation."""
        params = GenerateConfsParams(
            backend='openeye',
            max_confs=50,
            rms_threshold=0.5,
            SMILES_column='smiles',
            names_column='molecule_id',
            dropna=False,
            mode='classic'
        )

        actor = GenerateConfs(params, logger=logger)
        actor_input = ActorInput(data=test_data, context=pipeline_context)

        output = actor(actor_input)

        # Check output structure
        assert output.success
        assert 'n_conformers' in output.data.columns
        assert 'conformer_success' in output.data.columns

        # Check metadata
        assert output.metadata['backend'] == 'openeye'
        assert output.metadata['molecules_processed'] == 4

        # Check endpoint is file path
        assert isinstance(output.endpoint, str)
        assert output.endpoint.endswith('.oeb')

    @pytest.mark.skipif(
        not HAS_OPENEYE,
        reason="OpenEye toolkit not available"
    )
    def test_openeye_extract_molecules(self, test_data, logger, pipeline_context):
        """Test molecule extraction from OpenEye backend."""
        params = GenerateConfsParams(
            backend='openeye',
            max_confs=20,
            SMILES_column='smiles',
            names_column='molecule_id',
            dropna=True
        )

        actor = GenerateConfs(params, logger=logger)
        actor_input = ActorInput(data=test_data, context=pipeline_context)

        output = actor(actor_input)

        # Extract molecules
        molecules = list(actor.extract_molecules())

        # Should have successful molecules
        assert len(molecules) > 0

        # Check they are RDKit molecules with conformers
        for mol in molecules:
            assert isinstance(mol, Chem.Mol)
            assert mol.GetNumConformers() > 0


class TestBackendRegistry:
    """Tests for backend registry system."""

    def test_list_backends(self):
        """Test listing available backends."""
        from molforge.backends.registry import BackendRegistry

        backends = BackendRegistry.list_backends()
        assert 'confs' in backends.keys()

        backends = backends['confs']
        assert 'rdkit' in backends
        assert 'openeye' in backends

    def test_get_backend(self):
        """Test getting backend by name."""
        from molforge.backends.confs import RDKitBackend, OpenEyeBackend
        from molforge.backends.registry import BackendRegistry
        
        rdkit_cls = BackendRegistry.get('confs', 'rdkit')
        assert rdkit_cls == RDKitBackend

        openeye_cls = BackendRegistry.get('confs', 'openeye')
        assert openeye_cls == OpenEyeBackend

    def test_invalid_backend(self, test_data, logger, pipeline_context):
        """Test invalid backend name raises error."""
        with pytest.raises(ValueError, match="Unknown backend 'invalid_backend' for type 'confs'"):
            params = GenerateConfsParams(
                backend='invalid_backend',
                SMILES_column='smiles',
                names_column='molecule_id'
            )
            actor = GenerateConfs(params, logger=logger)
