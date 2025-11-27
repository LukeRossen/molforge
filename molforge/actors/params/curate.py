from typing import List, Union, Dict, Any, Literal
from dataclasses import dataclass, field

import os, shutil

from .base import BaseParams

@dataclass
class CurateMolParams(BaseParams):
    """Configuration parameters for molecule curation component."""

    mol_steps: List[str] = field(default_factory=lambda: [
        'desalt', 
        'removeIsotope', 
        'removeHs', 
        'tautomers',
        'neutralize', 
        'sanitize',
        'handleStereo', 
        'computeProps',
    ])
    # smiles steps are handled separately, and at the end of the pipe
    smiles_steps: List[str] = field(default_factory=lambda: [
        'canonical', 
        'kekulize', 
    ])
    
    neutralize_policy: Literal['keep', 'remove'] = 'keep'
    desalt_policy: Literal['keep', 'remove'] = 'keep'
    brute_force_desalt: bool = False
    max_tautomers: int = 512
    
    step_timeout: int = 60  # Limit a single step to 60s/mol (mostly for extremely large/complex molecules and CIP assignment)
            
    SMILES_column: str = 'canonical_smiles'
    dropna: bool = True
    check_duplicates: bool = True
    duplicates_policy: Literal['first', 'last', False] = 'first' # pandas drop_duplicates argument

    stereo_policy: Literal['keep', 'remove', 'assign', 'enumerate'] = 'assign'
    assign_policy: Literal['first', 'random', 'lowest'] = 'random'
    
    random_seed: int = 42    
    try_embedding: bool = False
    only_unassigned: bool = True
    only_unique: bool = True
    max_isomers: int = 32
    
    _VALID_MOL_STEPS: List[str] = field(default_factory=lambda: [
        'desalt', 'removeIsotope', 'removeHs', 'tautomers', 'neutralize', 'sanitize',
        'computeProps', 'handleStereo',
    ])
    _VALID_SMILES_STEPS: List[str] = field(default_factory=lambda: [
        'canonical', 'kekulize',
    ])
    _POSSIBLE_STEPS: List[str] = field(default_factory=lambda: [
        'desalt', 'removeIsotope', 'removeHs', 'tautomers', 'neutralize', 'sanitize',
        'computeProps', 'handleStereo', 'canonical', 'kekulize', 
    ])
    _BASIC_POLICIES: List[str] = field(default_factory=lambda: ['keep', 'remove'])
    _STEREO_POLICIES: List[str] = field(default_factory=lambda: ['assign', 'enumerate'])
    _ASSIGN_POLICIES: List[str] = field(default_factory=lambda: ['first', 'random', 'lowest'])
    
        
    def _validate_params(self) -> None:
        self._validate_policy('desalt_policy', self.desalt_policy, self._BASIC_POLICIES)
        self._validate_policy('neutralize_policy', self.neutralize_policy, self._BASIC_POLICIES)
        self._validate_policy('duplicates_policy', self.duplicates_policy, ['first', 'last', False])

        self._validate_policy('stereo_policy', self.stereo_policy, self._BASIC_POLICIES + self._STEREO_POLICIES)
        self._validate_policy('assign_policy', self.assign_policy, self._ASSIGN_POLICIES)       
        
        self._validate_steps()
    
    def _post_init_hook(self) -> None:
        self._configure_stereo_params()
        self._configure_pipe_steps()
    
    def _configure_pipe_steps(self):
        all_steps = self.mol_steps + self.smiles_steps
        
        for step in self._POSSIBLE_STEPS:
            setattr(self, step, False)
        
        for step in all_steps:
            if hasattr(self, step):
                setattr(self, step, True)
    
    def _configure_stereo_params(self) -> None:
        stereo_fields = [
            'stereo_policy', 'assign_policy', 'max_isomers', 
            'try_embedding', 'only_unassigned', 'only_unique', 'random_seed'
        ]
        self.stereo_params = {f: getattr(self, f) for f in stereo_fields}

    def _validate_steps(self) -> None:
        """Validate specified steps."""

        for step in self.mol_steps:
            if step not in self._VALID_MOL_STEPS:
                raise ValueError(f"Invalid mol_step '{step}'. Valid mol steps are: {self._VALID_MOL_STEPS}")
        
        for step in self.smiles_steps:
            if step not in self._VALID_SMILES_STEPS:
                raise ValueError(f"Invalid smiles_step '{step}'. Valid SMILES steps are: {self._VALID_SMILES_STEPS}")
        
        if len(self.mol_steps) != len(set(self.mol_steps)):
            duplicates = [step for step in set(self.mol_steps) if self.mol_steps.count(step) > 1]
            raise ValueError(f"Duplicate mol_steps found: {duplicates}")
            
        if len(self.smiles_steps) != len(set(self.smiles_steps)):
            duplicates = [step for step in set(self.smiles_steps) if self.smiles_steps.count(step) > 1]
            raise ValueError(f"Duplicate smiles_steps found: {duplicates}")
        
        if set(self.mol_steps) & set(self.smiles_steps):
            raise ValueError(f"Steps cannot be in both mol_steps and smiles_steps: {set(self.mol_steps) & set(self.smiles_steps)}")