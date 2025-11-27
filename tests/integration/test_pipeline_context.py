"""Integration tests for pipeline context and actor communication."""

import pytest
import pandas as pd
from pathlib import Path
from molforge import (
    ForgeParams, ConstructPipe,
    ChEMBLSourceParams, ChEMBLCuratorParams,
    CurateMolParams, TokenizeDataParams,
    GenerateConfsParams
)
from molforge.actors.protocol import ActorOutput
from molforge.configuration.context import PipelineContext
from molforge.configuration.steps import Steps

@pytest.fixture
def test_output_dir(tmp_path):
    """Create a temporary directory for test outputs."""
    output_dir = tmp_path / "pipeline_test"
    output_dir.mkdir()
    return str(output_dir)

@pytest.fixture
def context_test_params(test_output_dir):
    """Test configuration that exercises context communication."""
    return ForgeParams(
        verbose=True,
        override_actor_params=True,
        return_none_on_fail=False,  # Important: we want to see errors
        write_checkpoints=True,     # Test checkpoint writing
        output_root=test_output_dir,

        # Use steps that require context communication
        steps=['source', 'chembl', 'curate', 'tokens', 'confs'],

        # Configure each step
        source_params=ChEMBLSourceParams(
            backend='sql',
            auto_download=True,  # Will download ChEMBL database if not present
            version=36
        ),
        chembl_params=ChEMBLCuratorParams(
            standard_type='IC50',
            standard_units='nM',
            standard_relation='=',
            assay_type='B',
            target_organism='Homo sapiens',
            assay_format='protein',
            std_threshold=0.5,
            range_threshold=0.5,
        ),
        curate_params=CurateMolParams(
            SMILES_column='canonical_smiles',
            mol_steps=['desalt', 'removeIsotope', 'neutralize', 'sanitize', 'handleStereo'],
            smiles_steps=['canonical'],
            desalt_policy='keep',
            neutralize_policy='keep',
            stereo_policy='assign',
            dropna=True
        ),
        tokens_params=TokenizeDataParams(
            vocab_file=None,  # Will create dynamically
            SMILES_column='curated_smiles',
            dynamically_update_vocab=True
        ),
        # Add conformer generation with RDKit backend for faster testing
        confs_params=GenerateConfsParams(
            backend='rdkit',
            max_confs=10,  # Small number for testing
            rms_threshold=0.5,
            random_seed=42,  # For reproducibility
            use_uff=True,  # Optimize conformers
            max_iterations=200
        )
    )

def test_pipeline_context_initialization(context_test_params):
    """Test that pipeline properly initializes context."""
    pipeline = ConstructPipe(context_test_params)
    
    # Run pipeline
    result = pipeline('CHEMBL234')
    
    # Access pipeline's context
    context = pipeline.context
    
    assert isinstance(context, PipelineContext)
    assert context.get('input_id') == 'CHEMBL234'
    assert context.get('input_type') == 'ChEMBL ID'
    
    # Verify actors were registered
    assert context.get_actor(Steps.SOURCE) is not None
    assert context.get_actor(Steps.CHEMBL) is not None
    assert context.get_actor(Steps.CURATE) is not None
    assert context.get_actor(Steps.TOKENS) is not None
    assert context.get_actor(Steps.CONFS) is not None

def test_actor_communication(context_test_params):
    """Test that actors can communicate via context."""
    pipeline = ConstructPipe(context_test_params)
    result = pipeline('CHEMBL234')
    
    # Access pipeline's context
    context = pipeline.context
    
    # Test ChEMBL data source success tracking
    cs_result = context.get_actor_result(Steps.SOURCE)
    assert isinstance(cs_result, ActorOutput)
    assert cs_result.success is True
    assert 'n_activities' in cs_result.metadata
    assert cs_result.metadata['n_activities'] > 0

    # Check ChEMBL curator success tracking
    cc_result = context.get_actor_result(Steps.CHEMBL)
    assert isinstance(cc_result, ActorOutput)
    assert cc_result.success is True
    assert 'n_curated' in cc_result.metadata
    assert 'output_label' in cc_result.metadata
    assert 'standard_type' in cc_result.metadata
    assert cc_result.metadata['n_curated'] > 0

    # Check CurateMol success tracking
    cm_result = context.get_actor_result(Steps.CURATE)
    assert isinstance(cm_result, ActorOutput)
    assert cm_result.success is True
    assert 'n_curated' in cm_result.metadata or 'n_processed' in cm_result.metadata
    # CurateMol stores results in endpoint (file path), not metadata

    # Check TokenizeData success tracking and endpoint
    td_result = context.get_actor_result(Steps.TOKENS)
    assert isinstance(td_result, ActorOutput)
    assert td_result.success is True
    assert 'vocab_size' in td_result.metadata
    assert 'n_tokenized' in td_result.metadata
    assert td_result.endpoint is not None
    assert Path(td_result.endpoint).exists()

    # Check conformer generation success tracking
    gc_result = context.get_actor_result(Steps.CONFS)
    assert isinstance(gc_result, ActorOutput)
    assert gc_result.success is True
    assert gc_result.metadata['backend'] == 'rdkit'
    assert 'molecules_processed' in gc_result.metadata
    assert 'success_count' in gc_result.metadata
    assert 'total_conformers' in gc_result.metadata
    assert 'avg_conformers_per_molecule' in gc_result.metadata
    assert gc_result.metadata['success_count'] > 0
    assert gc_result.metadata['success_count'] <= gc_result.metadata['molecules_processed']

    # Verify conformers were generated
    gc_actor = context.get_actor(Steps.CONFS)
    mols = list(gc_actor.extract_molecules())
    assert len(mols) > 0
    assert all(mol.GetNumConformers() > 0 for mol in mols)

    # Verify sequential processing
    all_results = [cs_result, cc_result, cm_result, td_result, gc_result]
    assert all(result.success for result in all_results), "All actors should succeed"

def test_pipeline_error_handling(context_test_params):
    """Test that pipeline properly handles and reports errors."""
    # Test that params validation catches invalid database path
    with pytest.raises(FileNotFoundError) as exc_info:
        bad_params = ForgeParams(
            steps=context_test_params.steps,
            override_actor_params=context_test_params.override_actor_params,
            output_root=context_test_params.output_root,
            return_none_on_fail=False,  # Ensure errors are raised
            source_params=ChEMBLSourceParams(
                backend='sql',
                auto_download=False,  # Disable auto-download
                db_path="/nonexistent/path/chembl.db",  # Invalid path
                version=36
            ),
            chembl_params=context_test_params.chembl_params,
            curate_params=context_test_params.curate_params,
            tokens_params=context_test_params.tokens_params,
            confs_params=context_test_params.confs_params
        )

    # Check specific error messages
    error_msg = str(exc_info.value)
    assert "ChEMBL database not found at" in error_msg
    assert "/nonexistent/path/chembl.db" in error_msg
    assert "auto_download=True" in error_msg

def test_pipeline_checkpoints(context_test_params, test_output_dir):
    """Test that pipeline writes checkpoints correctly."""
    pipeline = ConstructPipe(context_test_params)
    result = pipeline('CHEMBL234')

    # Get the run ID from the pipeline
    run_id = pipeline.run_id

    # Check checkpoint files exist and verify their content
    # Note: output_dir is already the full path including run_id
    output_dir = Path(pipeline.output_dir)
    
    # ChEMBL source checkpoint verification
    cs_file = output_dir / f"{run_id}_source.csv"
    assert cs_file.exists()
    cs_df = pd.read_csv(cs_file)
    required_cs_cols = ['molecule_chembl_id', 'standard_value', 'canonical_smiles']
    assert all(col in cs_df.columns for col in required_cs_cols), \
        f"Missing columns in source output. Has: {list(cs_df.columns)[:10]}"
    assert len(cs_df) > 0
    
    # ChEMBL curator checkpoint
    chembl_file = output_dir / f"{run_id}_chembl.csv"
    assert chembl_file.exists()
    chembl_df = pd.read_csv(chembl_file)
    assert 'activity_id' in chembl_df.columns
    assert 'pIC50' in chembl_df.columns  # Output label column
    
    # Check other checkpoint files exist with basic content validation
    checkpoint_files = {
        'curate': ['curated_smiles', 'curation_success'],
        'tokens': ['tokens', 'seqlen'],
        'confs': ['conformer_success', 'n_conformers']  # Actual column name
    }
    
    for step, required_cols in checkpoint_files.items():
        file_path = output_dir / f"{run_id}_{step}.csv"
        assert file_path.exists(), f"Checkpoint file for {step} is missing"
        
        df = pd.read_csv(file_path)
        assert all(col in df.columns for col in required_cols), \
            f"Missing required columns in {step} checkpoint"
        assert len(df) > 0, f"Checkpoint file {step} is empty"

    # Verify checkpoint files were all created
    checkpoint_count = len(list(output_dir.glob(f"{run_id}_*.csv")))
    # Expect: source, chembl, curate, tokens, confs = 5 checkpoints
    assert checkpoint_count == 5, f"Expected 5 checkpoint files, found {checkpoint_count}"

def test_pipeline_state_consistency(context_test_params):
    """Test that pipeline maintains consistent state."""
    pipeline = ConstructPipe(context_test_params)
    
    # Run multiple times
    result1 = pipeline('CHEMBL234')
    context1 = pipeline.context
    
    result2 = pipeline('CHEMBL235')
    context2 = pipeline.context
    
    # Verify we got a fresh context
    assert context1 is not context2
    assert context1.get('input_id') == 'CHEMBL234'
    assert context2.get('input_id') == 'CHEMBL235'

def test_pipeline_output_validation(context_test_params):
    """Test that pipeline output contains expected data."""
    pipeline = ConstructPipe(context_test_params)
    result = pipeline('CHEMBL234')
    
    # Basic DataFrame checks
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    
    # Check columns from each step
    assert 'standard_value' in result.columns  # From ChEMBL source
    assert 'activity_id' in result.columns     # From ChEMBL curator
    assert 'curated_smiles' in result.columns  # From CurateMol
    assert 'tokens' in result.columns          # From TokenizeData
    assert 'conformer_success' in result.columns  # From GenerateConfs
    assert 'n_conformers' in result.columns      # From GenerateConfs (actual column name)

    # Check data validity
    assert not result['curated_smiles'].isnull().any()
    assert not result['tokens'].isnull().any()
    assert result['conformer_success'].sum() > 0  # At least some conformers generated
    assert (result['n_conformers'] > 0).sum() > 0  # Some molecules have conformers

if __name__ == '__main__':
    pytest.main([__file__, '-v'])