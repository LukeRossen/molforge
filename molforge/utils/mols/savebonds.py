
import json
from typing import Dict, List, Tuple, Union
from rdkit import Chem

import sqlite3
import os

class CanonicalBondProperties:
    """
    Semi-efficient handler for storing and retrieving bond properties using canonical atom rankings.
    Works with single molecules or batches of molecules. Originally written to use with torsion calculations.
    """
    
    @staticmethod
    def bond_idx_to_canonical(mol: Chem.Mol, bond_properties: Dict[int, float]) -> Dict[Tuple[int, int], float]:
        """
        Convert bond index mapping to canonical atom pair mapping.
        
        Args:
            mol: RDKit molecule object
            bond_properties: Dict mapping bond indices to property values
            
        Returns:
            Dict mapping canonical atom rank pairs to property values
        """
        if not bond_properties:
            return {}
            
        atom_ranks = Chem.CanonicalRankAtoms(mol)
        canonical_props = {}
        
        for bond_idx, value in bond_properties.items():
            
            bond = mol.GetBondWithIdx(bond_idx)
            
            rank1 = atom_ranks[bond.GetBeginAtomIdx()]
            rank2 = atom_ranks[bond.GetEndAtomIdx()]
            
            # Always use sorted tuple for consistent ordering
            canonical_key = tuple(sorted([rank1, rank2]))
            canonical_props[canonical_key] = value
            
        return canonical_props
    
    @staticmethod
    def canonical_to_bond_idx(mol: Chem.Mol, canonical_properties: Dict[Tuple[int, int], float]) -> Dict[int, float]:
        """
        Convert canonical atom pair mapping back to bond index mapping.
        
        Args:
            mol: RDKit molecule object  
            canonical_properties: Dict mapping canonical atom rank pairs to property values
            
        Returns:
            Dict mapping bond indices to property values
        """
        if not canonical_properties:
            return {}
            

        canonical_ranks = Chem.CanonicalRankAtoms(mol)
        rank_to_atom = {canonical_ranks[i]: i for i in range(len(canonical_ranks))}
        
        bond_props = {}
        
        for (rank1, rank2), value in canonical_properties.items():
            
            at1 = rank_to_atom[rank1]
            at2 = rank_to_atom[rank2]
            try:    
                bond_idx = mol.GetBondBetweenAtoms(at1, at2).GetIdx()
                bond_props[bond_idx] = value
    
            except AttributeError as e:
                print(f"Could not canonicalize bond properties for Molecule {Chem.MolToSmiles(mol)} | Error: {e}")
                return {}
           
     
        return bond_props
    
    @staticmethod
    def save_to_json(data: Union[Tuple[str, Dict], List[Tuple[str, Dict]]], filepath: str) -> None:
        """
        Save molecule(s) with canonical bond properties to JSON.
        
        Args:
            data: Either a single (smiles, canonical_properties) tuple or list of such tuples
            filepath: Output JSON file path
        """
        if isinstance(data, tuple):
            # Single molecule
            smiles, canonical_props = data
            json_data = {
                'smiles': smiles,
                'bond_properties': {f"{k[0]}-{k[1]}": v for k, v in canonical_props.items()}
            }
        else:
            # List of molecules
            json_data = []
            for smiles, canonical_props in data:
                mol_data = {
                    'smiles': smiles,
                    'bond_properties': {f"{k[0]}-{k[1]}": v for k, v in canonical_props.items()}
                }
                json_data.append(mol_data)
        
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    @staticmethod
    def load_from_json(filepath: str) -> Union[Tuple[str, Dict], List[Tuple[str, Dict]]]:
        """
        Load molecule(s) with canonical bond properties from JSON.
        
        Args:
            filepath: Input JSON file path
            
        Returns:
            Either a single (smiles, canonical_properties) tuple or list of such tuples
        """
        with open(filepath, 'r') as f:
            json_data = json.load(f)
        
        def parse_mol_data(mol_data):
            smiles = mol_data['smiles']
            canonical_props = {}
            for key_str, value in mol_data['bond_properties'].items():
                # Parse "rank1-rank2" back to tuple
                ranks = tuple(map(int, key_str.split('-')))
                canonical_props[ranks] = value
            return smiles, canonical_props
        
        if isinstance(json_data, dict):
            # Single molecule
            return parse_mol_data(json_data)
        else:
            # List of molecules
            return [parse_mol_data(mol_data) for mol_data in json_data]


    @staticmethod
    def add_molecule_to_file(mol: Chem.Mol, bond_properties: Dict[int, float], 
                        filepath: str, overwrite_existing: bool = False) -> None:
        """
        Add a single molecule with bond properties to a JSON file.
        
        Args:
            mol: RDKit molecule object
            bond_properties: Dict mapping bond indices to property values
            filepath: JSON file path (will be created if it doesn't exist)
            overwrite_existing: If True, replace existing entry with same SMILES. If False, skip duplicates.
        """
        # Generate canonical form for the new molecule
        smiles = Chem.MolToSmiles(mol, allHsExplicit=True)
        canonical_props = CanonicalBondProperties.bond_idx_to_canonical(mol, bond_properties)
        
        new_mol_data = {
            'smiles': smiles,
            'bond_properties': {f"{k[0]}-{k[1]}": v for k, v in canonical_props.items()}
        }
        
        # Load existing data or create empty list
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    existing_data = json.load(f)
                if isinstance(existing_data, dict):
                    existing_data = [existing_data]  # Convert single molecule to list
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = []
        else:
            existing_data = []
        
        # Check for existing SMILES
        existing_index = None
        for i, mol_data in enumerate(existing_data):
            if mol_data['smiles'] == smiles:
                existing_index = i
                break
        
        if existing_index is not None:
            if overwrite_existing:
                existing_data[existing_index] = new_mol_data
        else:
            existing_data.append(new_mol_data)
        
        # Save updated data
        with open(filepath, 'w') as f:
            json.dump(existing_data, f, indent=2)
            

    @staticmethod
    def canonicalize_molecule_batch(molecules: List[Chem.Mol], properties: List[Dict[int, float]]) -> List[Tuple[str, Dict]]:
        """
        Process a batch of molecules and their bond properties to canonical form.
        
        Args:
            molecules: List of molecules
            properties: List of corresponding bond properties
            
        Returns:
            List of (smiles, canonical_properties) tuples ready for JSON storage
        """
        canonical_data = []
        
        assert len(molecules) == len(properties)
        
        for mol, bond_props in zip(molecules, properties):
            # explicit Hs for consistent atom ranking.
            smiles = Chem.MolToSmiles(mol, allHsExplicit=True)
            canonical_props = CanonicalBondProperties.bond_idx_to_canonical(mol, bond_props)
            # TODO: add name
            canonical_data.append((smiles, canonical_props))
            
        return canonical_data


    @staticmethod
    def restore_molecule_batch(canonical_data: List[Tuple[str, Dict]]) -> Tuple[List[Chem.Mol], List[Dict[int, float]]]:
        """
        Restore a batch of molecules and bond properties from canonical form.
        
        Args:
            canonical_data: List of (smiles, canonical_properties) tuples
            
        Returns:
            List of molecules, List of corresponding bond_properties
        """
        molecules = []
        properties = []
        
        # explicit Hs for consistent atom ranking.
        params = Chem.SmilesParserParams()
        params.removeHs = False

        for smiles, canonical_props in canonical_data:
            mol = Chem.MolFromSmiles(smiles, params)
            if mol is not None:
                bond_props = CanonicalBondProperties.canonical_to_bond_idx(mol, canonical_props)
                # TODO: add name     
                molecules.append(mol)
                properties.append(bond_props)
                            
        return molecules, properties


    @staticmethod
    def save_molecules(molecules: List[Chem.Mol], properties: List[Dict[int, float]], filepath: str) -> None:
        """Canonicalize a batch of molecules with bond properties by atom ranking and save bond properties to json."""
        canonical_data = CanonicalBondProperties.canonicalize_molecule_batch(molecules, properties)
        CanonicalBondProperties.save_to_json(canonical_data, filepath)
        return
    
    @staticmethod
    def load_molecules(filepath: str) -> Tuple[List[Chem.Mol], List[Dict[int, float]]]:
        """Load a batch of molecules with bond properties from json and convert canonical atom ranking back to atom indexing."""
        canonical_data = CanonicalBondProperties.load_from_json(filepath)
        molecules, properties = CanonicalBondProperties.restore_molecule_batch(canonical_data)
        return  molecules, properties
        

class CanonicalBondPropertiesDB:
    
    @staticmethod
    def initialize_database(db_path: str) -> None:
        """Initialize database tables."""
        # TODO: allow overwrite to avoid duplicate appends for repeat calls.
        with sqlite3.connect(db_path) as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS molecules (
                    name TEXT PRIMARY KEY,
                    smiles TEXT NOT NULL
                );
                
                CREATE TABLE IF NOT EXISTS bond_properties (
                    name TEXT,
                    rank1 INTEGER,
                    rank2 INTEGER,
                    value REAL,
                    FOREIGN KEY (name) REFERENCES molecules (name)
                );
            ''')
    
    @staticmethod
    def add_molecule(db_path: str, mol: Chem.Mol, bond_properties: Dict[int, float], 
                    name: str, overwrite: bool = False) -> None:
        """Add molecule to database."""
        smiles = Chem.MolToSmiles(mol, allHsExplicit=True)
        canonical_props = CanonicalBondProperties.bond_idx_to_canonical(mol, bond_properties)
        
        with sqlite3.connect(db_path) as conn:
            if overwrite:
                conn.execute('DELETE FROM molecules WHERE name = ?', (name,))
                conn.execute('DELETE FROM bond_properties WHERE name = ?', (name,))
            
            conn.execute('INSERT OR IGNORE INTO molecules (name, smiles) VALUES (?, ?)', (name, smiles))
            
            bond_data = [(name, rank1, rank2, value) for (rank1, rank2), value in canonical_props.items()]
            conn.executemany('INSERT INTO bond_properties (name, rank1, rank2, value) VALUES (?, ?, ?, ?)', bond_data)


    @staticmethod
    def add_molecules(db_path: str, molecules: List[Chem.Mol], properties: List[Dict[int, float]], names: List[str]) -> None:
        """Add multiple molecules to database."""
        assert len(molecules) == len(properties) == len(names)
        
        for mol, bond_props, name in zip(molecules, properties, names):
            CanonicalBondPropertiesDB.add_molecule(db_path, mol, bond_props, name)
            
    @staticmethod
    def load_all_molecules(db_path: str) -> Tuple[List[Chem.Mol], List[Dict[int, float]], List[str]]:
        """Load all molecules and properties."""
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute('''
                SELECT m.name, m.smiles, bp.rank1, bp.rank2, bp.value
                FROM molecules m
                JOIN bond_properties bp ON m.name = bp.name
                ORDER BY m.name
            ''').fetchall()
            
            molecules, properties, names = [], [], []
            params = Chem.SmilesParserParams()
            params.removeHs = False
            
            current_name = None
            current_canonical_props = {}
            
            for name, smiles, rank1, rank2, value in rows:
                if name != current_name:
                    if current_name is not None:
                        mol = Chem.MolFromSmiles(current_smiles, params)
                        bond_props = CanonicalBondProperties.canonical_to_bond_idx(mol, current_canonical_props)
                        molecules.append(mol)
                        properties.append(bond_props)
                        names.append(current_name)
                    
                    current_name = name
                    current_smiles = smiles
                    current_canonical_props = {}
                
                current_canonical_props[(rank1, rank2)] = value
            
            # Last molecule
            if current_name is not None:
                mol = Chem.MolFromSmiles(current_smiles, params)
                bond_props = CanonicalBondProperties.canonical_to_bond_idx(mol, current_canonical_props)
                molecules.append(mol)
                properties.append(bond_props)
                names.append(current_name)
            
            for mol, name in zip(molecules, names):
                mol.SetProp("_Name", name)
            
            return molecules, properties