# Fast integration test for pipeline (chembl_source -> chembl -> curate -> tokens)

import pytest
import pandas as pd
from molforge import (
    ForgeParams, ConstructPipe,
    ChEMBLSourceParams, ChEMBLCuratorParams,
    CurateMolParams, TokenizeDataParams
)

@pytest.fixture
def output_dir(tmp_path):
    """Unified output directory for all test artifacts."""
    out_dir = tmp_path / "test_output"
    out_dir.mkdir(exist_ok=True)
    return out_dir

@pytest.fixture
def fast_params(output_dir):
    """Fast test configuration (minimal steps) with unified output dir."""
    return ForgeParams(
        verbose=True,  # Quiet for tests
        override_actor_params=True,
        return_none_on_fail=True,
        write_checkpoints=False,
        output_root=str(output_dir),

        steps=['source', 'chembl', 'curate', 'tokens'],

        curate_params=CurateMolParams(
            SMILES_column='canonical_smiles',
            mol_steps=['desalt', 'removeIsotope', 'neutralize', 'sanitize', 'handleStereo'],
            smiles_steps=['canonical'],
            desalt_policy='keep',
            neutralize_policy='keep',
            stereo_policy='assign',
            dropna=True,
        ),
        tokens_params=TokenizeDataParams(),
        source_params=ChEMBLSourceParams(
            backend='sql',
            auto_download=True,  # Will download ChEMBL database if not present
            version=36
        ),
    )

def test_pipeline_fast_execution(fast_params):
    """Test full pipeline executes without errors."""
    pipeline = ConstructPipe(fast_params)

    # Use small target for fast test
    result = pipeline('CHEMBL234')

    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert 'tokens' in result.columns  # From tokens step
    assert 'curated_smiles' in result.columns  # From curate step
    assert len(result) > 0

if __name__ == '__main__':
    test_pipeline_fast_execution(fast_params())