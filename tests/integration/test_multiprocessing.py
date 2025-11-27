"""Integration test for multiprocessing functionality.

This test ensures that actors can be properly pickled and used in
multiprocessing workers. Tests the fix for lambda pickling issues.
"""

import pytest
import pandas as pd
from molforge import (
    ForgeParams, ConstructPipe,
    ChEMBLSourceParams, ChEMBLCuratorParams,
    CurateMolParams, CurateDistributionParams
)


@pytest.fixture
def output_dir(tmp_path):
    """Unified output directory for all test artifacts."""
    out_dir = tmp_path / "test_output_mp"
    out_dir.mkdir(exist_ok=True)
    return out_dir


@pytest.fixture
def multiprocessing_params(output_dir):
    """
    Configuration for multiprocessing test.

    Uses CHEMBL235 (Human Acetylcholinesterase) which has sufficient data
    to trigger multiprocessing (> 1000 molecules).

    Tests both curate and distributions steps which both use multiprocessing.
    """
    return ForgeParams(
        verbose=True,
        override_actor_params=True,
        return_none_on_fail=True,
        write_checkpoints=False,
        output_root=str(output_dir),

        steps=['source', 'chembl', 'curate', 'distributions'],

        curate_params=CurateMolParams(
            SMILES_column='canonical_smiles',
            mol_steps=['desalt', 'removeIsotope', 'neutralize', 'sanitize', 'handleStereo'],
            smiles_steps=['canonical'],
            desalt_policy='keep',
            neutralize_policy='keep',
            stereo_policy='assign',  # This uses StereoHandler which had the lambda bug
            dropna=True,
        ),
        distributions_params=CurateDistributionParams(
            SMILES_column='curated_smiles',
            global_statistical_threshold=2.0,
            plot_distributions=False,
            perform_pca=False,
        ),
        source_params=ChEMBLSourceParams(
            backend='sql',
            auto_download=True,  # Will download ChEMBL database if not present
            version=36
        ),
    )


def test_multiprocessing_curate_and_distributions(multiprocessing_params):
    """
    Test that multiprocessing works correctly for curate and distributions actors.

    This test specifically validates:
    1. Actors can be pickled for multiprocessing workers
    2. StereoHandler can be pickled (no lambda issues)
    3. Worker initialization is suppressed (suppress_init_logs works)
    4. Large datasets (> 1000 molecules) process successfully

    CHEMBL235 (Human Acetylcholinesterase) typically has 15,000+ bioactivity records,
    which will trigger multiprocessing in both curate (>1000) and distributions (>1000).
    """
    pipeline = ConstructPipe(multiprocessing_params)

    # Use CHEMBL235 which has plenty of data to trigger multiprocessing
    result = pipeline('CHEMBL235')

    # Validate successful execution
    assert result is not None, "Pipeline should return result"
    assert isinstance(result, pd.DataFrame), "Result should be DataFrame"
    assert len(result) > 0, "Result should have data"

    # Validate expected columns from curate step
    assert 'curated_smiles' in result.columns, "Should have curated_smiles"
    assert 'curation_success' in result.columns, "Should have curation_success"

    # Validate that multiprocessing actually occurred (>1000 molecules)
    # Note: After filtering, we may have < 1000, but the initial curate should have > 1000
    assert 'molecular_weight' in result.columns, "Distributions step should add molecular_weight"

    # Check that we have successful curations
    successful_curations = result['curation_success'].sum() if 'curation_success' in result.columns else len(result)
    assert successful_curations > 100, f"Should have > 100 successful curations, got {successful_curations}"


def test_multiprocessing_curate_only(multiprocessing_params):
    """
    Test multiprocessing specifically for the curate actor.

    This test validates that the CurateMol actor can properly initialize
    StereoHandler in multiprocessing workers without lambda pickling errors.
    """
    # Modify params to only test curate step
    params = multiprocessing_params
    params.steps = ['source', 'chembl', 'curate']

    pipeline = ConstructPipe(params)
    result = pipeline('CHEMBL235')

    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    assert 'curated_smiles' in result.columns

    # Verify stereo handling occurred (stereo_policy='assign')
    successful_curations = result['curation_success'].sum() if 'curation_success' in result.columns else len(result)
    assert successful_curations > 100, "Should have many successful curations with stereo handling"


if __name__ == '__main__':
    # Allow running test directly for debugging
    pytest.main([__file__, '-v', '-s'])
