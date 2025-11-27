"""
RDKit ETKDG conformer generation backend.

Uses RDKit's ETKDG (Experimental Torsion Knowledge Distance Geometry) method
for conformer generation. Stores molecules in memory for efficient processing.
"""

from typing import Dict, Any, Iterator, Tuple
import time
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from rdkit import Chem
from rdkit.Chem import AllChem

from .base import ConformerBackend
from ...utils.actortools.multiprocess import calculate_chunk_params


def _generate_conformer_chunk(
    chunk: list[Tuple[str, str]],
    max_confs: int,
    random_seed: int,
    rms_threshold: float,
    use_random_coords: bool,
    use_uff: bool,
    max_iterations: int
) -> list[Tuple[str, Dict[str, Any]]]:
    """
    Generate conformers for a chunk of SMILES.

    Module-level function for efficient parallel processing with chunking.
    Processes multiple molecules in a single worker to reduce overhead.

    Args:
        chunk: List of (smiles, name) tuples
        max_confs: Maximum number of conformers to generate
        random_seed: Random seed for reproducibility
        rms_threshold: RMS threshold for conformer pruning
        use_random_coords: Whether to use random coordinates
        use_uff: Whether to use UFF optimization
        max_iterations: Maximum iterations for UFF optimization

    Returns:
        List of (name, result_dict) tuples for each molecule in chunk
    """
    results = []
    for smiles, name in chunk:
        result = _generate_single_conformer(
            smiles, name,
            max_confs, random_seed, rms_threshold,
            use_random_coords, use_uff, max_iterations
        )
        results.append(result)
    return results


def _generate_single_conformer(
    smiles: str,
    name: str,
    max_confs: int,
    random_seed: int,
    rms_threshold: float,
    use_random_coords: bool,
    use_uff: bool,
    max_iterations: int
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate conformers for a single SMILES using RDKit ETKDG.

    Module-level function for efficient parallel processing.
    Does not access instance attributes, making it safe for multiprocessing.

    Args:
        smiles: SMILES string
        name: Molecule identifier
        max_confs: Maximum number of conformers to generate
        random_seed: Random seed for reproducibility
        rms_threshold: RMS threshold for conformer pruning
        use_random_coords: Whether to use random coordinates
        use_uff: Whether to use UFF optimization
        max_iterations: Maximum iterations for UFF optimization

    Returns:
        Tuple of (name, result_dict) where result_dict contains:
            - success: bool
            - n_conformers: int
            - status: str
            - error: str or None
            - molecule: Chem.Mol (if successful)
            - rotors: int
            - elapsed_time: float
    """
    start_time = time.time()

    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return (name, {
                'success': False,
                'n_conformers': 0,
                'status': '',
                'error': 'Invalid SMILES',
                'rotors': 0,
                'elapsed_time': time.time() - start_time
            })

        # Add hydrogens
        mol = Chem.AddHs(mol)
        mol.SetProp("_Name", name)

        # Count rotatable bonds
        rotors = Chem.Lipinski.NumRotatableBonds(mol)

        # Configure ETKDG parameters
        etkdg_params = AllChem.ETKDGv3()
        etkdg_params.randomSeed = random_seed
        etkdg_params.numThreads = 1  # Each worker processes one molecule
        etkdg_params.pruneRmsThresh = rms_threshold
        etkdg_params.useRandomCoords = use_random_coords

        # Generate conformers
        conf_ids = AllChem.EmbedMultipleConfs(
            mol,
            numConfs=max_confs,
            params=etkdg_params
        )

        if len(conf_ids) == 0:
            return (name, {
                'success': False,
                'n_conformers': 0,
                'status': '',
                'error': 'Conformer embedding failed',
                'rotors': rotors,
                'elapsed_time': time.time() - start_time
            })

        # Optional UFF optimization
        if use_uff:
            for conf_id in conf_ids:
                try:
                    AllChem.UFFOptimizeMolecule(
                        mol,
                        confId=conf_id,
                        maxIters=max_iterations
                    )
                except Exception:
                    pass  # Continue if optimization fails

        elapsed = time.time() - start_time

        return (name, {
            'success': True,
            'n_conformers': len(conf_ids),
            'status': '',  # Empty like OMEGA successful generations
            'error': None,
            'molecule': mol,
            'rotors': rotors,
            'elapsed_time': round(elapsed, 2)
        })

    except Exception as e:
        return (name, {
            'success': False,
            'n_conformers': 0,
            'status': '',
            'error': str(e),
            'rotors': 0,
            'elapsed_time': time.time() - start_time
        })


class RDKitBackend(ConformerBackend):
    """
    RDKit ETKDG conformer generation backend.

    Features:
    - In-memory storage (no file I/O)
    - ETKDG conformer generation
    - Optional UFF optimization
    - Configurable pruning threshold
    """
    __backend_name__ = 'RDK'

    # Backend metadata
    description = "Generate conformers using RDKit ETKDGv3 algorithm"
    required_params = ['SMILES_column']
    optional_params = [
        'names_column', 'max_confs', 'n_jobs', 'rms_threshold',
        'random_seed', 'verbose', 'dropna'
    ]

    def __init__(self, params, logger=None, context=None):
        """Initialize RDKit backend."""
        super().__init__(params, logger, context)

        # Storage for generated conformers (in input order)
        self._molecules: list[Chem.Mol] = []
        self._report_data: list[Dict[str, Any]] = []

        self.log(
            f"RDKit ETKDG initialized: max_confs={params.max_confs}, "
            f"rms={params.rms_threshold}, optimize={params.use_uff}"
        )

    def generate_conformers(self, smiles_list: list[str], names_list: list[str]) -> Dict[str, Dict[str, Any]]:
        """
        Generate conformers using RDKit ETKDG with parallel processing and progress reporting.

        Args:
            smiles_list: List of SMILES strings
            names_list: List of molecule identifiers

        Returns:
            Dict mapping names to generation results
        """
        # Clear previous results
        self._molecules.clear()
        self._report_data.clear()

        # Prepare chunks (use smaller chunk size for conformer generation)
        max_chunk_size = 50  # Smaller chunks enable frequent progress updates
        n_processes, chunk_size, _ = calculate_chunk_params(len(smiles_list), max_chunk_size)

        # Create chunks of (smiles, name) pairs
        pairs = list(zip(smiles_list, names_list))
        chunks = [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]
        n_chunks = len(chunks)

        self.log(
            f"Generating conformers for {len(smiles_list):,} molecules with {n_processes} processes "
            f"({n_chunks} chunks of {chunk_size})."
        )

        # Process chunks in parallel
        results_dict = {}
        chunk_results = [None] * n_chunks
        start_time = time.time()
        completed_chunks = 0

        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            # Submit all chunk tasks
            future_to_chunk = {
                executor.submit(
                    _generate_conformer_chunk,
                    chunk,
                    self.params.max_confs,
                    self.params.random_seed,
                    self.params.rms_threshold,
                    self.params.use_random_coords,
                    self.params.use_uff,
                    self.params.max_iterations
                ): i
                for i, chunk in enumerate(chunks)
            }

            # Process as completed with progress reporting
            for future in as_completed(future_to_chunk, timeout=3600):
                chunk_idx = future_to_chunk[future]
                try:
                    chunk_results[chunk_idx] = future.result(timeout=300)
                    completed_chunks += 1

                    # Report progress after each chunk
                    self._report_progress(completed_chunks, n_chunks, start_time, chunk_size)

                except Exception as e:
                    self.log(f"Chunk {chunk_idx} failed: {e}", level='ERROR')
                    # Fill with failures for this chunk
                    chunk_results[chunk_idx] = [
                        (name, {
                            'success': False,
                            'n_conformers': 0,
                            'status': '',
                            'error': f'Chunk processing error: {str(e)}',
                            'rotors': 0,
                            'elapsed_time': 0.0
                        })
                        for _, name in chunks[chunk_idx]
                    ]
                    completed_chunks += 1

        # Flatten results into dict
        for chunk_result in chunk_results:
            if chunk_result:
                for name, result in chunk_result:
                    results_dict[name] = result

        # Store molecules and build report in input order
        for smiles, name in zip(smiles_list, names_list):
            result = results_dict[name]

            # Build report entry (mimics OMEGA format)
            self._report_data.append({
                'Molecule': name,
                'SMILES': smiles,
                'Rotors': result.get('rotors', 0),
                'Conformers': result['n_conformers'],
                'ElapsedTime(s)': result.get('elapsed_time', 0.0),
                'Status': result['status']
            })

            # Store successful molecules
            if result['success'] and 'molecule' in result:
                mol = result['molecule']
                # Ensure name property is set (may be lost during pickling/unpickling)
                if not mol.HasProp('_Name'):
                    mol.SetProp('_Name', name)
                self._molecules.append(mol)

        success_count = sum(1 for r in results_dict.values() if r['success'])
        self.log(f"RDKit generation complete: {success_count}/{len(smiles_list)} succeeded")

        return results_dict

    def _report_progress(self, completed_chunks: int, total_chunks: int,
                         start_time: float, chunk_size: int):
        """
        Report conformer generation progress.

        Args:
            completed_chunks: Number of chunks completed
            total_chunks: Total number of chunks
            start_time: Start time of generation
            chunk_size: Approximate molecules per chunk
        """
        elapsed = time.time() - start_time
        progress_pct = 100 * completed_chunks / total_chunks
        molecules_processed = completed_chunks * chunk_size

        if completed_chunks > 0:
            estimated_total = elapsed * total_chunks / completed_chunks
            eta = estimated_total - elapsed
            rate = molecules_processed / elapsed

            self.log(
                f"Progress: {completed_chunks}/{total_chunks} chunks "
                f"({progress_pct:.1f}%) | "
                f"Elapsed: {elapsed:.1f}s | "
                f"ETA: {eta:.1f}s | "
                f"Rate: {rate:.0f} mol/s"
            )

    def get_endpoint(self) -> list[Chem.Mol]:
        """
        Return list of molecules as endpoint.

        For in-memory backends, return the stored list.
        """
        return self._molecules

    def extract_molecules(self) -> Iterator[Chem.Mol]:
        """
        Extract molecules in generation order.

        Yields:
            RDKit molecules with conformers
        """
        for mol in self._molecules:
            yield mol

    def get_report_dataframe(self) -> pd.DataFrame:
        """Get conformer generation report as DataFrame (mimics OMEGA format)."""
        import pandas as pd
        if not self._report_data:
            return pd.DataFrame(columns=['Molecule', 'SMILES', 'Rotors', 'Conformers', 'ElapsedTime(s)', 'Status'])
        return pd.DataFrame(self._report_data)

    def get_successful_names(self) -> list[str]:
        """Get list of successfully generated molecule names."""
        return [mol.GetProp('_Name') for mol in self._molecules if mol.HasProp('_Name')]

    def clear(self):
        """Clear stored molecules to free memory."""
        self._molecules.clear()
        self._report_data.clear()
