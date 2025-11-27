# ============================================================================
# Example: Real Plugin Actor for Property Calculation
# ============================================================================

from dataclasses import dataclass
from typing import Any
import pandas as pd

from molforge.actors.base import BaseActor
from molforge.actors.params import BaseParams

@dataclass
class CalculatePropertiesParams(BaseParams):
    """Parameters for property calculation."""
    
    properties: list = None  # Which properties to calculate
    SMILES_column: str = 'curated_smiles'
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = ['MolWt', 'MolLogP', 'TPSA']
        super().__post_init__()

    def _validate_params(self) -> None:
        # validate params must be manually defined, to ensure users perform sanity checks.
        from rdkit.Chem import Descriptors

        for prop_name in self.properties:
            if not hasattr(Descriptors, prop_name):
                raise AttributeError(f"Property with name {prop_name} does not exist in rdkit.Chem.Descriptors.")
            
    
class CalculateProperties(BaseActor):
    """Calculate molecular descriptors and add as columns."""
    
    __plugin_name__ = 'properties'
    __param_class__ = CalculatePropertiesParams
    __extra_kwargs__ = {}
    
    def __init__(self, params: CalculatePropertiesParams, logger=None, **kwargs):
        self.method = '[PROPS]'
        self._params = params
        
        for key, value in params.to_dict().items():
            setattr(self, key, value)
        
        self._setup_logging(logger)
    
    def __call__(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate properties for molecules."""
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        
        self.print(f"Calculating {len(self.properties)} properties")
        
        df = input_data.copy()
        
        # Calculate each property
        for prop_name in self.properties:
            if hasattr(Descriptors, prop_name):
                descriptor_func = getattr(Descriptors, prop_name)
                
                values = []
                for smiles in df[self.SMILES_column]:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        values.append(descriptor_func(mol))
                    else:
                        values.append(None)
                
                df[prop_name] = values
                self.print(f"  Calculated {prop_name}")
        
        return df