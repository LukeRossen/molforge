"""
Torsion analysis actor plugin.

Analyzes torsional properties for molecules with conformers.
Requires phd-tools package for calculations.
"""

from typing import List, Tuple, Dict, Optional, Literal, Iterator, Any
import pickle
from pathlib import Path
import json
import time
import gc

import pandas as pd
from rdkit import Chem
from joblib import Parallel, delayed

# Attempt to import private toolkit
try:
    from phd_tools.chemistry.torsion import TorsionCalculator
    from phd_tools.chemistry.savebonds import CanonicalBondProperties
    TORSION_AVAILABLE = True
except ImportError:
    TORSION_AVAILABLE = False
    TorsionCalculator = None
    CanonicalBondProperties = None


# Plugin imports
from molforge.actors.base import BaseActor
from molforge.actors.protocol import ActorOutput
from molforge.actors.params.base import BaseParams
from molforge.utils.constants import MAX_CHUNK_SIZE, DEFAULT_MP_THRESHOLD, DEFAULT_N_JOBS

from dataclasses import dataclass


# ============================================================================
# PARAMETERS
# ============================================================================

@dataclass
class CalculateTorsionsParams(BaseParams):
    """
    Configuration for torsion analysis.

    Parameters control computational settings and analysis thresholds.
    See phd-tools documentation for detailed parameter descriptions.
    """

    aggregation_method: Literal['min', 'max', 'mean'] = 'mean'
    account_for_symmetry: bool = True
    symmetry_radius: int = 3
    ignore_colinear_bonds: bool = True
    n_confs_threshold: int = 50
    min_confs: Optional[int] = None
    """Hard minimum conformer count. Molecules below this are treated as
    failures and removed from results. None disables this filter.
    Distinct from n_confs_threshold, which only flags an informational warning."""
    dropna: bool = True
    """Drop rows where torsion analysis failed. Failures include: torsion
    computation errors (malformed molecule, no torsions), SMILES reconstruction
    failures, bond canonicalization errors, and min_confs violations.
    Rows without upstream conformers are also dropped."""

    def _validate_params(self) -> None:
        """Validate parameter values."""
        self._validate_policy(
            'aggregation_method',
            self.aggregation_method,
            ['min', 'max', 'mean']
        )

        if self.n_confs_threshold < 1:
            raise ValueError("Conformer threshold must be at least 1")

        if self.min_confs is not None and self.min_confs < 1:
            raise ValueError("min_confs must be at least 1")


# ============================================================================
# MODULE-LEVEL FUNCTIONS (pickling-safe for joblib workers)
# ============================================================================

def _default_row(name: str = '') -> Dict:
    """Return a complete default result row matching OUTPUT_COLUMNS."""
    return {
        'torsion_name': name,
        'torsion_smiles': '',
        'torsion_mapping': '{}',
        'torsion_success': False,
        'n_confs': 0,
        'low_confs': False,
        'n_ring_torsions': 0,
        'n_rotor_torsions': 0,
        'n_torsions': 0,
        'mean_variance': 0.0,
        'warnings': [],
    }


def _serialize_mapping(mapping: Dict) -> str:
    """Serialize canonical mapping to JSON for CSV-safe storage."""
    if not mapping:
        return '{}'
    return json.dumps({
        'canonical_atom_ranks': [list(k) for k in mapping.keys()],
        'circular_variance': list(mapping.values()),
    })


def _mark_failed(row: Dict, warning: str) -> None:
    """Mark a result row as failed, clearing molecule-specific fields."""
    row['torsion_success'] = False
    row['torsion_smiles'] = ''
    row['torsion_mapping'] = '{}'
    row['warnings'].append(warning)


def _process_batch_worker(
    batch: List[Tuple[str, Any]],
    calculator_config: Dict,
) -> List[Dict]:
    """
    Worker function for parallel batch processing.

    Returns row dicts matching OUTPUT_COLUMNS, plus '_canonical_mapping'
    sidecar for post-worker bond index reconstruction.
    """
    # Prevent PyTorch OpenMP thread oversubscription in forked workers
    import torch
    torch.set_num_threads(1)

    calculator = TorsionCalculator(**calculator_config)

    results = []
    for name, mol in batch:
        if mol is None:
            results.append(_default_row(name))
            continue

        try:
            bond_variance, stats, rd_topology = calculator.compute(mol)
            canonical_mapping = CanonicalBondProperties.bond_idx_to_canonical(
                rd_topology, bond_variance
            )
            torsion_smiles = Chem.MolToSmiles(rd_topology)

            row = {
                'torsion_name': name,
                'torsion_smiles': torsion_smiles,
                'torsion_mapping': _serialize_mapping(canonical_mapping),
                'torsion_success': True,
                '_canonical_mapping': canonical_mapping,
                **stats,
            }
            results.append(row)

        except Exception as e:
            row = _default_row(name)
            row['warnings'] = [f"Torsion computation: {type(e).__name__}: {str(e)}"]
            results.append(row)

    return results


# ============================================================================
# ACTOR
# ============================================================================

class CalculateTorsions(BaseActor):
    """
    Torsion analysis actor.

    Analyzes torsional properties for molecules with multiple conformers.
    Outputs bond-level metrics and conformer statistics.

    Uses chunked parallel processing for large datasets. Results are saved
    incrementally to avoid memory issues with millions of molecules.

    Requires:
        - GenerateConfs actor output (conformers)
        - phd-tools package (private)

    Output columns:
        - torsion_name: Molecule name/identifier (derived from confs actor)
        - torsion_smiles: SMILES with explicit hydrogens for molecule reconstruction
        - torsion_mapping: JSON with 'canonical_atom_ranks' and 'circular_variance'
        - torsion_success: Boolean flag for successful torsion calculation
        - n_confs: Number of conformers analyzed
        - n_torsions: Total torsions detected
        - n_rotor_torsions: Rotatable bond torsions
        - n_ring_torsions: Ring torsions
        - mean_variance: Average metric across bonds
        - low_confs: Flag for low conformer count
        - warnings: Analysis warnings
    """

    __step_name__ = 'torsion'
    __param_class__ = CalculateTorsionsParams

    # Single source of truth for the added columns: the default result row.
    OUTPUT_COLUMNS = list(_default_row())

    @property
    def required_columns(self) -> List[str]:
        """Required columns from GenerateConfs."""
        return ['conformer_success']

    @property
    def output_columns(self) -> List[str]:
        """Columns added by torsion analysis."""
        return self.OUTPUT_COLUMNS

    @property
    def forge_endpoint(self) -> str:
        """Endpoint for MolForge integration. Points to torsion mapping column."""
        return 'torsion_mapping'

    @property
    def torsions_dir(self) -> Path:
        """Path to torsions directory in current run directory."""
        dir_path = Path(self._get_run_path("torsions"))
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    def __post_init__(self):
        """Initialize torsion analysis."""
        if not TORSION_AVAILABLE:
            raise ImportError(
                "This actor requires the 'phd-tools' package.\n"
                "\n"
                "Installation:\n"
                "  pip install git+ssh://git@github.com/LukeRossen/phd-tools.git\n"
                "\n"
                "Note: Repository access required. Contact maintainer."
            )

        # Store calculator config for worker processes
        self._calculator_config = {
            'aggregation_method': self.aggregation_method,
            'account_for_symmetry': self.account_for_symmetry,
            'symmetry_radius': self.symmetry_radius,
            'ignore_colinear_bonds': self.ignore_colinear_bonds,
            'n_confs_threshold': self.n_confs_threshold,
        }

        self.log(
            f"Torsion analysis initialized "
            f"(workers: {DEFAULT_N_JOBS}, chunk_size: {MAX_CHUNK_SIZE:,})"
        )

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze torsional properties for molecules with conformers.

        Processes molecules in parallel chunks to handle large datasets
        efficiently while managing memory. Uses name-based merging to
        guarantee correct alignment regardless of DataFrame index state.

        Args:
            data: DataFrame with conformer_success and Title columns

        Returns:
            DataFrame with torsion analysis columns added
        """
        df = data.copy()

        gc_actor = self.get_actor('confs')
        if gc_actor is None:
            raise ValueError(
                "GenerateConfs actor not found. "
                "Ensure GenerateConfs runs before CalculateTorsions."
            )

        # Get the title column name from confs actor (the join key)
        title_col = gc_actor.get_title_column()
        if title_col not in df.columns:
            raise ValueError(
                f"Expected column '{title_col}' from confs actor not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

        # Intake summary
        n_with_confs = int(df['conformer_success'].sum())
        n_without_confs = len(df) - n_with_confs
        self.log(
            f"Received {len(df):,} molecules "
            f"({n_with_confs:,} with conformers, {n_without_confs:,} without)."
        )

        # Get molecules and their names from confs actor (both in same order)
        mol_generator = gc_actor.extract_molecules()
        names = gc_actor.get_successful_names()

        # Process only successful molecules, keyed by name
        torsion_results_df = self._process_chunked(mol_generator, names)

        # Merge results back to DataFrame by name — guarantees correct alignment
        df = pd.merge(
            df, torsion_results_df,
            left_on=title_col, right_on='torsion_name',
            how='left'
        )

        # Fill NaN for molecules without conformers (not processed)
        df = self._fill_empty_torsion_cols(df)

        # Report: torsion-specific failures (not upstream)
        failed = df[~df['torsion_success'] & df['conformer_success']]
        if len(failed) > 0:
            failure_types = failed['warnings'].apply(
                lambda w: w[0].split(':')[0] if w else 'Unknown'
            )
            summary = failure_types.value_counts()
            self.log(
                f"Failure summary ({len(failed):,} torsion failures):\n"
                f"{summary.to_frame('count').to_markdown()}",
                level='WARNING'
            )

        # Report: low conformer warnings (informational, not failures)
        successful = df[df['torsion_success']]
        n_low_confs = int(successful['low_confs'].sum()) if len(successful) > 0 else 0
        if n_low_confs > 0:
            self.log(
                f"{n_low_confs:,} successful molecules flagged with low conformer "
                f"count (< {self.n_confs_threshold}). Results may be less "
                f"statistically reliable for these molecules."
            )

        # Outcome funnel
        n_success = len(successful)
        n_removed = len(df) - n_success
        pct_removed = (100 * n_removed / len(df)) if len(df) else 0.0
        self.log(
            f"Torsion analysis: {len(df):,} → {n_success:,} succeeded "
            f"(removed {n_removed:,}, {pct_removed:.1f}%)  |  "
            f"no conformers {n_without_confs:,}, failed {len(failed):,}, "
            f"low conformers {n_low_confs:,}"
        )

        # Drop failed rows if requested
        if self.dropna:
            initial_count = len(df)
            df = df[df['torsion_success']]
            dropped = initial_count - len(df)
            if dropped > 0:
                self.log(f"Dropped {dropped:,} rows with failed torsion analysis")

        return df

    def _process_chunked(
        self,
        molecules: Iterator,
        names: List[str],
    ) -> pd.DataFrame:
        """
        Process molecules in chunks with optional parallelism.

        Uses single-process mode for small datasets (< DEFAULT_MP_THRESHOLD)
        and parallel processing for larger ones.

        Args:
            molecules: Generator yielding successful molecules (same order as names)
            names: List of molecule names from confs actor (success only)

        Returns:
            DataFrame with torsion results keyed by torsion_name
        """
        total_mols = len(names)
        use_mp = total_mols >= DEFAULT_MP_THRESHOLD
        total_chunks = (total_mols + MAX_CHUNK_SIZE - 1) // MAX_CHUNK_SIZE

        # Dispatch log
        if use_mp:
            self.log(
                f"Processing {total_mols:,} molecules "
                f"with {DEFAULT_N_JOBS} workers ({total_chunks} chunks of {MAX_CHUNK_SIZE:,})."
            )
        else:
            self.log(
                f"Processing {total_mols:,} molecules in single process."
            )

        chunk_idx = 0
        processed_count = 0
        chunk_data = []
        pipeline_start = time.time()
        chunk_times = []
        manifest = []
        all_results = []

        mol_iter = zip(names, molecules)

        for i, (name, mol) in enumerate(mol_iter):
            chunk_data.append((name, mol))

            # Process chunk when full or at end
            if len(chunk_data) >= MAX_CHUNK_SIZE or i == total_mols - 1:
                chunk_start = time.time()
                chunk_results, manifest_entry = self._process_chunk(
                    chunk_data, chunk_idx, use_mp
                )
                chunk_time = time.time() - chunk_start
                chunk_times.append(chunk_time)

                if manifest_entry is not None:
                    manifest.append(manifest_entry)

                all_results.extend(chunk_results)
                processed_count += len(chunk_data)

                # Per-chunk progress
                elapsed = time.time() - pipeline_start
                rate = processed_count / elapsed if elapsed > 0 else 0
                self.log(
                    f"Chunk {chunk_idx + 1}/{total_chunks} | "
                    f"{processed_count:,}/{total_mols:,} "
                    f"({100 * processed_count / total_mols:.1f}%) | "
                    f"Rate: {rate:.0f} mol/s | Chunk: {chunk_time:.1f}s"
                )

                chunk_data = []
                chunk_idx += 1
                gc.collect()

        # Write manifest
        if manifest:
            self._write_manifest(manifest)
            total_named = sum(e['count'] for e in manifest)
            self.log(f"Wrote manifest: {len(manifest)} chunks, {total_named:,} molecules")

        # Completion summary
        total_time = time.time() - pipeline_start
        avg_chunk = sum(chunk_times) / len(chunk_times) if chunk_times else 0
        self.log(
            f"Completed: {processed_count:,} molecules in {chunk_idx} chunks "
            f"({total_time:.1f}s total, avg {avg_chunk:.1f}s/chunk)"
        )

        return pd.DataFrame(all_results)

    def _process_chunk(
        self,
        chunk_data: List[Tuple[str, Any]],
        chunk_idx: int,
        use_mp: bool = True,
    ) -> Tuple[List[Dict], Optional[Dict[str, Any]]]:
        """
        Process a chunk of molecules, optionally in parallel.

        Worker returns row dicts directly. This method validates successful
        results (min_confs, SMILES round-trip, bond canonicalization) and
        accumulates pickle data for chunk saving.

        Args:
            chunk_data: List of (name, molecule) tuples
            chunk_idx: Chunk index for logging/saving
            use_mp: Whether to use multiprocessing

        Returns:
            Tuple of (list of row dicts matching OUTPUT_COLUMNS, manifest entry or None)
        """
        if use_mp:
            batches = self._split_into_batches(chunk_data, DEFAULT_N_JOBS)
            batch_results = Parallel(n_jobs=DEFAULT_N_JOBS)(
                delayed(_process_batch_worker)(batch, self._calculator_config)
                for batch in batches
            )
        else:
            batch_results = [
                _process_batch_worker(chunk_data, self._calculator_config)
            ]

        # Collect worker row dicts keyed by name
        processed_results = {}
        for batch_result in batch_results:
            for row in batch_result:
                processed_results[row['torsion_name']] = row

        # Validate successful results and accumulate pickle data
        rows = []
        chunk_molecules = []
        chunk_mappings = []

        smiles_params = Chem.SmilesParserParams()
        smiles_params.removeHs = False

        for name, _ in chunk_data:
            row = processed_results.get(name, _default_row(name))
            canonical_mapping = row.pop('_canonical_mapping', None)

            if row['torsion_success']:
                if self.min_confs is not None and row['n_confs'] < self.min_confs:
                    _mark_failed(row, f"Minimum conformers: {row['n_confs']} < {self.min_confs}")
                else:
                    mol_obj = Chem.MolFromSmiles(row['torsion_smiles'], smiles_params)
                    if mol_obj is None:
                        _mark_failed(row, 'SMILES reconstruction: MolFromSmiles returned None')
                    else:
                        try:
                            bond_map = CanonicalBondProperties.canonical_to_bond_idx(
                                mol_obj, canonical_mapping
                            )
                            mol_obj.SetProp('_Name', name)
                            chunk_molecules.append(mol_obj)
                            chunk_mappings.append(bond_map)
                        except (ValueError, KeyError) as e:
                            _mark_failed(row, f"Bond canonicalization: {e}")

            rows.append(row)

        # Save chunk to pickle
        manifest_entry = None
        if chunk_molecules:
            manifest_entry = self._save_chunk(
                chunk_molecules, chunk_mappings, chunk_idx
            )

        return rows, manifest_entry

    @staticmethod
    def _fill_empty_torsion_cols(df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill NaN values in torsion columns for rows that weren't processed.

        Rows without conformers won't have torsion results after the merge.
        """
        defaults = _default_row()
        for col, default in defaults.items():
            if col not in df.columns:
                continue
            if col == 'warnings':
                df[col] = df[col].apply(
                    lambda x: x if isinstance(x, list) else []
                )
            else:
                df[col] = df[col].fillna(default)

        df['torsion_success'] = df['torsion_success'].astype(bool)
        return df

    @staticmethod
    def _split_into_batches(
        items: List[Any],
        n_batches: int,
    ) -> List[List[Any]]:
        """
        Split items into n approximately equal batches.

        Args:
            items: List of items to split
            n_batches: Number of batches to create

        Returns:
            List of batches (lists)
        """
        if not items:
            return []

        # Ensure at least 1 batch
        n_batches = max(1, min(n_batches, len(items)))
        batch_size = (len(items) + n_batches - 1) // n_batches
    
        return [
            items[i:i + batch_size]
            for i in range(0, len(items), batch_size)
        ]

    def _save_chunk(
        self,
        molecules: List[Chem.Mol],
        mappings: List[Dict],
        chunk_idx: int,
    ) -> Dict[str, Any]:
        """
        Save chunk results to pickle file.

        Args:
            molecules: List of RDKit molecules
            mappings: List of bond variance mappings
            chunk_idx: Chunk index for filename

        Returns:
            Manifest entry dict with file, names, and count
        """
        filename = f"chunk_{chunk_idx:04d}.pkl"
        chunk_path = self.torsions_dir / filename

        names = [
            mol.GetProp('_Name') if mol.HasProp('_Name') else ''
            for mol in molecules
        ]

        with open(chunk_path, 'wb') as f:
            pickle.dump((molecules, mappings), f)

        return {'file': filename, 'names': names, 'count': len(molecules)}

    def _write_manifest(self, manifest: List[Dict[str, Any]]) -> None:
        """
        Write manifest file for fast name lookup and integrity checking.

        Args:
            manifest: List of chunk entries with file, names, count
        """
        manifest_path = self.torsions_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

    def _create_output(self, data: pd.DataFrame) -> ActorOutput:
        """Create output with torsion mapping endpoint."""
        if 'n_torsions' in data.columns and 'torsion_success' in data.columns:
            success_df = data[data['torsion_success']]
            total_torsions = success_df['n_torsions'].sum() if len(success_df) > 0 else 0
            mean_metric = success_df['mean_variance'].mean() if len(success_df) > 0 else 0.0
            n_success = len(success_df)
        else:
            total_torsions = 0
            mean_metric = 0.0
            n_success = 0

        manifest_path = self.torsions_dir / "manifest.json"
        n_chunks = len(json.loads(manifest_path.read_text())) if manifest_path.exists() else 0

        return ActorOutput(
            data=data,
            success=True,
            metadata={
                'n_molecules': len(data),
                'n_success': n_success,
                'total_torsions': int(total_torsions),
                'mean_metric': float(mean_metric),
                'n_chunks': n_chunks,
            },
            endpoint=self.forge_endpoint
        )
