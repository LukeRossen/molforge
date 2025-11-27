"""
MolForge - Clean API for molecular processing pipeline.

Forges raw molecular data through multi-step processing (API retrieval,
curation, tokenization, conformer generation) into refined molecules
ready for storage in MolBox.
"""

from typing import Union, Optional, Tuple, Dict, Any, Iterable
from pathlib import Path
import pandas as pd

from .configuration.pipe import ConstructPipe
from .configuration.factory import PipeParams
from .configuration.registry import ActorRegistry

try:
    import molbox
    from molbox import MolBox
    HAS_MOLBOX = True
except ImportError:
    try:
        from ...molbox import MolBox
        HAS_MOLBOX = True
    except (ImportError, ValueError):
        HAS_MOLBOX = False


class MolForge:
    """
    Clean API for molecular processing pipeline.
    
    MolForge processes raw molecular data through configurable steps:
    - API retrieval (ChEMBL)
    - Data curation and filtering
    - Molecule curation and standardization
    - Tokenization
    - Conformer generation
    - Property computation
    
    Results can be saved as CSV or directly to MolBox for efficient storage.
    """
    
    def __init__(self,
                 params: Union[PipeParams, str, Dict] = None,
                 config_path: str = None,
                 output_root: Optional[str] = None,
                 **config_overrides):
        """
        Initialize MolForge pipeline.

        Args:
            params: PipeParams object, or dict of parameters
            config_path: Path to configuration file (YAML/JSON)
            output_root: Root output directory for INIT log and run subdirectories
            **config_overrides: Override specific config parameters

        Examples:
            >>> # From params object
            >>> forge = MolForge(params)

            >>> # From config file
            >>> forge = MolForge(config_path="pipeline_config.yaml")

            >>> # With custom root output directory
            >>> forge = MolForge(params, output_root="projects/kinases")

            >>> # With overrides
            >>> forge = MolForge(params, verbose=True, n_jobs=8)
        """
        # Handle different initialization methods
        if params is None and config_path is None:
            raise ValueError("Must provide either params or config_path")

        if config_path is not None:
            params = self._load_config(config_path)

        if isinstance(params, dict):
            params = PipeParams(**params)

        # Apply output_root override if provided
        if output_root is not None:
            config_overrides['output_root'] = output_root

        # Apply overrides
        if config_overrides:
            params_dict = params.to_dict()
            params_dict.update(config_overrides)
            params = PipeParams(**params_dict)

        # Initialize internal pipeline
        self._pipe = ConstructPipe(params)
        self._params = params
    
    def forge(self,
              input_data: Union[str, pd.DataFrame],
              input_id: Optional[str] = None,
              return_info: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, bool, str]]:
        """
        Forge molecules through the pipeline.

        Args:
            input_data: ChEMBL target ID (str), CSV file path (str), or DataFrame
            input_id: Optional identifier (auto-detected for ChEMBL/CSV)
            return_info: If True, return (dataframe, success, failed_step) tuple

        Returns:
            Processed DataFrame, or tuple if return_info=True

        Examples:
            >>> # From ChEMBL (output to configured output_root)
            >>> df = forge.forge("CHEMBL123")
            >>> # Output: <output_root>/CHEMBL123_IDabc/...

            >>> # From DataFrame
            >>> df = forge.forge(input_df, input_id="my_compounds")
            >>> # Output: <output_root>/my_compounds_IDabc/...

            >>> # With step info
            >>> df, success, step = forge.forge("CHEMBL123", return_info=True)
        """
        return self._pipe(input_data, input_id, return_info=return_info)
    
    def forge_to_csv(self,
                     input_data: Union[str, pd.DataFrame],
                     output_path: str,
                     input_id: Optional[str] = None) -> Tuple[pd.DataFrame, bool]:
        """
        Forge molecules and save to CSV.

        Args:
            input_data: ChEMBL target ID, CSV path, or DataFrame
            output_path: Path for output CSV file
            input_id: Optional identifier

        Returns:
            Tuple of (dataframe, success)

        Note:
            Pipeline intermediate files are written to output_root (set at MolForge initialization).
            Only the final CSV is written to output_path.

        Example:
            >>> df, success = forge.forge_to_csv("CHEMBL123", "output.csv")
        """
        df, success, _ = self._pipe(input_data, input_id, return_info=True)
        
        if success and df is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
        
        return df, success
    
    def forge_and_box(self,
                      input_data: Union[str, pd.DataFrame],
                      molbox_path: str,
                      input_id: Optional[str] = None,
                      smiles_column: str = None,
                      **molbox_kwargs) -> Tuple[pd.DataFrame, str]:
        """
        Forge molecules and save directly to MolBox.

        This is the recommended workflow for large datasets with conformers,
        as MolBox provides efficient storage and retrieval.

        Args:
            input_data: ChEMBL target ID, CSV path, or DataFrame
            molbox_path: Path for MolBox parquet database
            input_id: Optional identifier
            smiles_column: Column containing SMILES (default: from config)
            **molbox_kwargs: Additional arguments for MolBox (e.g., bond_properties, n_jobs)

        Returns:
            Tuple of (dataframe, molbox_path)

        Raises:
            ImportError: If molecule_database package not available

        Note:
            Pipeline intermediate files are written to output_root (set at MolForge initialization).
            Only the final MolBox parquet is written to molbox_path.

        Examples:
            >>> # Basic usage
            >>> df, path = forge.forge_and_box("CHEMBL123", "molecules.parquet")

            >>> # With properties
            >>> df, path = forge.forge_and_box(
            ...     "compounds.csv",
            ...     "molecules.parquet",
            ...     energy=energy_list,
            ...     n_jobs=8
            ... )
        """
        if not HAS_MOLBOX:
            raise ImportError(
                "molecule_database package required for forge_and_box. "
                "Use forge() or forge_to_csv() instead."
            )

        # Process through pipeline
        df, success, failed_step = self._pipe(input_data, input_id, return_info=True)
        
        if not success or df is None:
            raise RuntimeError(f"Pipeline failed at step: {failed_step}")
        
        # Convert DataFrame to molecules for MolBox
        molecules = self._dataframe_to_molecules(df, smiles_column)

        # Extract DataFrame properties (excluding MolBox internal columns)
        property_columns = [c for c in df.columns if c not in MolBox.INTERNAL_COLUMNS]
        properties = {col: df[col].tolist() for col in property_columns}

        # Merge with any additional molbox_kwargs properties (kwargs take precedence)
        properties.update(molbox_kwargs)

        # Save to MolBox with all properties
        MolBox.save_molecules(
            molecules,
            molbox_path,
            **properties
        )
        
        return df, molbox_path
    
    def batch_forge(self,
                    input_list: list,
                    output_dir: str,
                    use_molbox: bool = False,
                    **kwargs) -> Dict[str, Tuple[bool, str]]:
        """
        Forge multiple inputs in batch.
        
        Args:
            input_list: List of inputs (ChEMBL IDs, CSV paths, or DataFrames)
            output_dir: Directory for outputs
            use_molbox: If True, save to MolBox instead of CSV
            **kwargs: Additional arguments for forge_and_box or forge_to_csv
            
        Returns:
            Dict mapping input identifiers to (success, output_path) tuples
            
        Example:
            >>> results = forge.batch_forge(
            ...     ["CHEMBL123", "CHEMBL456"],
            ...     "output_dir",
            ...     use_molbox=True
            ... )
        """
        results = {}
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for input_data in input_list:
            # Generate identifier
            if isinstance(input_data, str):
                input_id = Path(input_data).stem if input_data.endswith('.csv') else input_data
            else:
                input_id = f"batch_{len(results)}"
            
            try:
                if use_molbox:
                    out_file = output_path / f"{input_id}.parquet"
                    df, path = self.forge_and_box(input_data, str(out_file), input_id, **kwargs)
                    success = df is not None
                else:
                    out_file = output_path / f"{input_id}.csv"
                    df, success = self.forge_to_csv(input_data, str(out_file), input_id)
                    path = str(out_file)
                
                results[input_id] = (success, path)
                
            except Exception as e:
                self._pipe.print(f"Failed to process {input_id}: {e}", level="ERROR")
                results[input_id] = (False, str(e))
        
        return results
    
    def _dataframe_to_molecules(self,
                                df: pd.DataFrame,
                                smiles_column: Optional[str] = None):
        """
        Convert DataFrame to molecule objects for MolBox storage.

        If conformers were generated in the pipeline, loads them from file.
        Otherwise, creates molecules from SMILES.

        Args:
            df: DataFrame with SMILES column and optionally conformer metadata
            smiles_column: Column name containing SMILES

        Returns:
            List of molecule objects (RDKit or OpenEye with conformers if available)
        """
        # Check if conformers were generated in this pipeline run
        
        attr_name = ActorRegistry.get_molecule_provider_attr(self.steps)

        if attr_name is not None:
            # return molecule_provider.forge_endpoint if molecule endpoints were generated.
            endpoint = getattr(self._pipe, attr_name).forge_endpoint
            
            if Path(endpoint).is_file() or (isinstance(endpoint, Iterable) and not isinstance(endpoint, str)):
                self._pipe.print(f"Inferring Molecular endpoint from '{attr_name}' ({type(endpoint)}).", level='INFO')
                return endpoint
            
            # if an endpoint simply points to a result column, we use that. 
            elif isinstance(endpoint, str) and endpoint in df.columns:
                self._pipe.print(f"Inferring SMILES endpoint from '{endpoint}'.", level='INFO')
                if not 'smiles' in endpoint.lower():
                    self._pipe.print(f"Column name '{endpoint}' is not obviously a SMILES column", level='WARN')
                smiles_column = endpoint
            
        # construct molecules from smiles if no molecules or conformers were generated.
        return self._molecules_from_smiles(df, smiles_column)

    def _molecules_from_smiles(self,
                               df: pd.DataFrame,
                               smiles_column: Optional[str] = None):
        """
        Create molecule objects from SMILES strings in DataFrame.

        Args:
            df: DataFrame with SMILES column
            smiles_column: Column name containing SMILES

        Returns:
            List of RDKit molecule objects without conformers
        """
        from rdkit import Chem

        # Get the most recent SMILES column if not specified, assuming last appended if the final product.
        if smiles_column is None:
            for col in df.columns[::-1]:
                if 'smiles' in col.lower():
                    smiles_column = col
                    self._pipe.print(f"No SMILES column specified, using last appended SMILES column: {smiles_column}.", level="WARNING")
                    break
                
            if smiles_column is None:   
                raise ValueError(f"SMILES column '{smiles_column}' not found in DataFrame")
                            

        if smiles_column not in df.columns:
            raise ValueError(f"SMILES column '{smiles_column}' not found in DataFrame")

        # Convert to molecules
        molecules = []
        for idx, row in df.iterrows():
            smiles = row[smiles_column]
            mol = Chem.MolFromSmiles(smiles)

            if mol is not None:
                # Set molecule name from DataFrame if available
                if 'name' in df.columns:
                    mol.SetProp("_Name", str(row['name']))
                else:
                    mol.SetProp("_Name", f"mol_{idx}")

                molecules.append(mol)
            else:
                self._pipe.print(f"Warning: Failed to parse SMILES at index {idx}", level="WARNING")

        return molecules
    
    def _load_config(self, config_path: str) -> PipeParams:
        """Load configuration from file."""
        import json
        import yaml
        
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load based on extension
        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
        
        return PipeParams(**config_dict)
    
    @property
    def config(self) -> PipeParams:
        """Access current configuration."""
        return self._params
    
    @property
    def steps(self) -> list:
        """Get configured pipeline steps."""
        return self._pipe.steps
    
    def get_actor(self, identifier: str) -> Any:
        """
        Retrieve an initialized actor from the pipe.

        Args:
            identifier: Step name (e.g., 'confs', 'curate') or attr_name (e.g., 'GC', 'CM')

        Returns:
            Actor instance or None if not found

        Note:
            This method now delegates to ActorRegistry.get_actor_instance()
            for centralized actor retrieval logic.
        """
        return ActorRegistry.get_actor_instance(self._pipe, identifier)
    
    def __repr__(self) -> str:
        steps_str = ' → '.join(self.steps)
        return f"MolForge({steps_str})"