"""
Flexibility analysis actor plugin.

Analyzes flexibility properties for molecules with conformers.
Requires phd-tools package for calculations.
"""
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import math
import os
import pickle
import subprocess
import time

import pandas as pd
from joblib import Parallel, delayed

try:
    from phd_tools.chemistry.flexibility import FlexibilityCalculator
    from phd_tools.chemistry.savebonds import (key_torsions_by_position, parse_smiles,
                                               torsion_values_at)
    FLEX_AVAILABLE = True
except ImportError:
    FLEX_AVAILABLE = False
    FlexibilityCalculator = None
    key_torsions_by_position = parse_smiles = torsion_values_at = None

from molforge.actors.base import BaseActor
from molforge.actors.protocol import ActorOutput
from molforge.actors.params.base import BaseParams
from molforge.utils.constants import MAX_CHUNK_SIZE, DEFAULT_MP_THRESHOLD, DEFAULT_N_JOBS

ENDPOINTS = ("V", "S")
FORMAT_VERSION = 2


# ============================================================================
# PARAMETERS
# ============================================================================
@dataclass
class CalculateFlexibilityParams(BaseParams):
    """Configuration for flexibility analysis.

    Parameters control computational settings and analysis thresholds.
    See phd-tools documentation for detailed parameter descriptions."""
    tau: float = 0.80
    n_confs_threshold: int = 50
    symmetry_radius: int = 3
    ignore_colinear_bonds: bool = True
    min_confs: Optional[int] = None
    dropna: bool = True
    chunk_size: int = MAX_CHUNK_SIZE
    """Molecules buffered per chunk before dispatch to workers. Defaults to molforge's
    MAX_CHUNK_SIZE (50k). Molecules arrive as full conformer ensembles, so peak RAM scales
    with chunk_size x conformers/mol; lower it for high max_confs (e.g. ~15k at 400 confs)
    to keep buffered poses near the ~10M proven-safe point."""

    def _validate_params(self) -> None:
        if not (0.0 <= self.tau <= 1.0):
            raise ValueError("tau must be in [0, 1]")
        if self.n_confs_threshold < 1:
            raise ValueError("n_confs_threshold must be at least 1")
        if self.min_confs is not None and self.min_confs < 1:
            raise ValueError("min_confs must be at least 1")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")


# ============================================================================
# MODULE-LEVEL FUNCTIONS (pickling-safe for joblib workers)
# ============================================================================
def _default_row(name: str = "") -> Dict:
    return {
        "flexibility_name": name, "flexibility_smiles": "", "flexibility_mapping": "{}",
        "flexibility_success": False, "n_confs": 0, "n_bonds": 0,
        "mean_V": 0.0, "low_confs": False, "warnings": [],
    }


def _mark_failed(row: Dict, warning: str) -> None:
    row["flexibility_success"] = False
    row["flexibility_smiles"] = ""
    row["flexibility_mapping"] = "{}"
    row["warnings"].append(warning)


def _code_identity(module) -> Dict[str, Any]:
    directory = os.path.dirname(os.path.abspath(module.__file__))

    def git(*args) -> Optional[str]:
        try:
            out = subprocess.run(["git", "-C", directory, *args], capture_output=True, text=True,
                                 timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": git("rev-parse", "HEAD"),
            "uncommitted_changes": None if status is None else bool(status)}


def _serialize_mapping(torsions: Dict) -> str:
    """Serialize mapping to JSON for CSV-safe storage."""
    if not torsions:
        return "{}"
    keys = list(torsions)
    return json.dumps({
        "torsion_atom_positions": [list(k) for k in keys],
        "V": [torsions[k]["V"] for k in keys], "S": [torsions[k]["S"] for k in keys],
    })


def _process_batch_worker(batch: List, config: Dict) -> List[Dict]:
    """Process a (name, mol) batch. Returns rows + per-endpoint sidecars (CPU per worker)."""
    import torch
    torch.set_num_threads(1)
    min_confs = config.get("_min_confs")
    calc = FlexibilityCalculator(device="cpu",
                                 **{k: v for k, v in config.items() if not k.startswith("_")})
    rows = []
    for name, mol in batch:
        if mol is None:
            rows.append(_default_row(name)); continue
        try:
            labels, stats, rd = calc.compute(mol)
            if min_confs is not None and stats["n_confs"] < min_confs:
                r = _default_row(name)
                r["warnings"] = [f"Minimum conformers: {stats['n_confs']} < {min_confs}"]
                rows.append(r); continue
            smiles, torsions = key_torsions_by_position(
                rd, {rec["torsion_atoms"]: {e: rec[e] for e in ENDPOINTS} for rec in labels.values()})
            rows.append({
                "flexibility_name": name,
                "flexibility_smiles": smiles,
                "flexibility_mapping": _serialize_mapping(torsions),
                "flexibility_success": True,
                "_torsion_positions": torsions,
                "n_confs": stats["n_confs"], "n_bonds": stats["n_bonds"],
                "mean_V": stats["mean_V"], "low_confs": stats["low_confs"], "warnings": [],
            })
        except Exception as e:
            r = _default_row(name)
            r["warnings"] = [f"Flexibility computation: {type(e).__name__}: {e}"]
            rows.append(r)
    return rows


# ============================================================================
# ACTOR
# ============================================================================
class CalculateFlexibility(BaseActor):
    """Flexibility actor. Requires GenerateConfs output + phd-tools."""

    __step_name__ = "flexibility"
    __param_class__ = CalculateFlexibilityParams
    OUTPUT_COLUMNS = list(_default_row())

    @property
    def required_columns(self) -> List[str]:
        return ["conformer_success"]

    @property
    def output_columns(self) -> List[str]:
        return self.OUTPUT_COLUMNS

    @property
    def forge_endpoint(self) -> str:
        return "flexibility_mapping"

    def endpoint_dir(self, endpoint: str) -> Path:
        d = Path(self._get_run_path("torsions")) / endpoint
        d.mkdir(parents=True, exist_ok=True)
        return d

    def __post_init__(self):
        if not FLEX_AVAILABLE:
            raise ImportError("This actor requires the private 'phd-tools' package "
                              "(phd_tools.chemistry.flexibility.FlexibilityCalculator).")
        self._calc_config = {
            "tau": self.tau, "n_confs_threshold": self.n_confs_threshold,
            "symmetry_radius": self.symmetry_radius,
            "ignore_colinear_bonds": self.ignore_colinear_bonds,
            "_min_confs": self.min_confs,
        }
        self.log(f"Flexibility analysis initialized (endpoints: {', '.join(ENDPOINTS)}, "
                 f"workers: {DEFAULT_N_JOBS}, chunk_size: {self.chunk_size:,})")

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        gc_actor = self.get_actor("confs")
        if gc_actor is None:
            raise ValueError("GenerateConfs actor not found. Run confs before flexibility.")
        title_col = gc_actor.get_title_column()
        if title_col not in df.columns:
            raise ValueError(f"Expected '{title_col}' from confs actor not in DataFrame.")

        n_with = int(df["conformer_success"].sum())
        self.log(f"Received {len(df):,} molecules ({n_with:,} with conformers).")
        self._write_provenance()

        results = self._process_chunked(gc_actor.extract_molecules(),
                                        gc_actor.get_successful_names())
        df = pd.merge(df, results, left_on=title_col, right_on="flexibility_name", how="left")
        df = self._fill_empty(df)

        failed = df[~df["flexibility_success"] & df["conformer_success"]]
        succ = df[df["flexibility_success"]]
        self.log(f"Flexibility: {len(df):,} → {len(succ):,} succeeded "
                 f"(failed {len(failed):,}, no-confs {len(df)-n_with:,}); "
                 f"sidecars → torsions/{{{','.join(ENDPOINTS)}}}/")
        if self.dropna:
            df = df[df["flexibility_success"]]
        return df

    def _process_chunked(self, mol_generator, names: List[str]) -> pd.DataFrame:
        total = len(names)
        use_mp = total >= DEFAULT_MP_THRESHOLD
        rows: List[Dict] = []
        manifests: Dict[str, List[Dict]] = {e: [] for e in ENDPOINTS}
        chunk: List = []
        chunk_idx = 0
        start_time = time.time()
        last_log = start_time
        for i, (name, mol) in enumerate(zip(names, mol_generator)):
            chunk.append((name, mol))
            if len(chunk) >= self.chunk_size or i == total - 1:
                chunk_rows, entries = self._process_chunk(chunk, chunk_idx, use_mp)
                rows.extend(chunk_rows)
                # Rewrite each endpoint manifest after its chunk so the index on disk
                # always matches the chunks written — an interruption leaves a valid,
                # builder-readable partial rather than orphaned, unindexed chunks.
                for e in ENDPOINTS:
                    if entries[e] is not None:
                        manifests[e].append(entries[e])
                        self._write_manifest(e, manifests[e])
                last_log = self._log_progress(total, start_time, last_log)
                chunk = []; chunk_idx += 1
        return pd.DataFrame(rows) if rows else pd.DataFrame([_default_row()]).iloc[:0]

    def _count_written(self, endpoint: str) -> int:
        """Molecules written to an endpoint folder so far, read from its manifest on disk."""
        manifest_path = self.endpoint_dir(endpoint) / "manifest.json"
        if not manifest_path.exists():
            return 0
        try:
            with open(manifest_path) as f:
                return sum(entry.get("count", 0) for entry in json.load(f))
        except (json.JSONDecodeError, OSError):
            return 0

    def _log_progress(self, total: int, start_time: float, last_log_time: float) -> float:
        """Read an endpoint folder and log progress, throttled by an ETA-adaptive interval
        (mirrors the confs progress monitor). Returns the updated last-log time."""
        done = self._count_written(ENDPOINTS[0])
        now = time.time()
        elapsed = now - start_time
        rate = done / elapsed if elapsed > 0 else 0.0
        if rate > 0 and done < total:
            eta = (total - done) / rate
            interval = min(max(5, int(10 ** (math.log10(max(eta, 1)) - 1))), 1000)
        else:
            eta, interval = 0.0, 5
        if done < total and (now - last_log_time) < interval:
            return last_log_time
        pct = (done / total * 100) if total else 100.0
        self.log(f"Progress: {done:,}/{total:,} ({pct:.1f}%) | "
                 f"Rate: {rate:.1f} mol/s | ETA: {self._format_duration(eta)}")
        return now

    def _process_chunk(self, chunk_data: List, chunk_idx: int, use_mp: bool
                       ) -> Tuple[List[Dict], Dict[str, Optional[Dict]]]:
        if use_mp and len(chunk_data) >= DEFAULT_N_JOBS:
            batches = [chunk_data[j::DEFAULT_N_JOBS] for j in range(DEFAULT_N_JOBS)]
            batch_results = Parallel(n_jobs=DEFAULT_N_JOBS)(
                delayed(_process_batch_worker)(b, self._calc_config) for b in batches if b)
        else:
            batch_results = [_process_batch_worker(chunk_data, self._calc_config)]

        processed = {row["flexibility_name"]: row for br in batch_results for row in br}
        rows, names, mols, torsions = [], [], [], []
        maps = {e: [] for e in ENDPOINTS}
        for name, _ in chunk_data:
            row = processed.get(name, _default_row(name))
            positions = row.pop("_torsion_positions", None)
            if row["flexibility_success"]:
                mol_obj = parse_smiles(row["flexibility_smiles"])
                if mol_obj is None:
                    _mark_failed(row, "SMILES reconstruction: MolFromSmiles returned None")
                else:
                    try:
                        torsion_atoms, values = torsion_values_at(mol_obj, positions)
                        bond_maps = {e: {b: v[e] for b, v in values.items()} for e in ENDPOINTS}
                        names.append(name)
                        mols.append(mol_obj)
                        torsions.append(torsion_atoms)
                        for e in ENDPOINTS:
                            maps[e].append(bond_maps[e])
                    except (ValueError, KeyError) as error:
                        _mark_failed(row, f"Label transfer: {error}")
            rows.append(row)

        entries = {e: None for e in ENDPOINTS}
        if mols:
            for e in ENDPOINTS:
                entries[e] = self._save_chunk(e, names, mols, maps[e], torsions, chunk_idx)
        return rows, entries

    def _save_chunk(self, endpoint: str, names: List[str], molecules: List, mappings: List,
                    torsion_atoms: List, chunk_idx: int) -> Dict:
        filename = f"chunk_{chunk_idx:04d}.pkl"
        with open(self.endpoint_dir(endpoint) / filename, "wb") as f:
            pickle.dump({"format_version": FORMAT_VERSION, "names": names, "molecules": molecules,
                         "values": mappings, "torsion_atoms": torsion_atoms}, f)
        return {"file": filename, "names": names, "count": len(molecules),
                "format_version": FORMAT_VERSION}

    def _write_provenance(self) -> None:
        import molforge
        import phd_tools
        import rdkit
        record = {
            "format_version": FORMAT_VERSION,
            "endpoints": list(ENDPOINTS),
            "parameters": {**{k: v for k, v in self._calc_config.items() if not k.startswith("_")},
                           "min_confs": self.min_confs, "chunk_size": self.chunk_size},
            "code": {"molforge": _code_identity(molforge), "phd_tools": _code_identity(phd_tools),
                     "rdkit": rdkit.__version__},
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        path = Path(self._get_run_path("torsions")) / "provenance.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(record, f, indent=2)
        os.replace(tmp, path)

    def _write_manifest(self, endpoint: str, manifest: List[Dict]) -> None:
        """Atomically replace the endpoint manifest (temp file + os.replace) so a crash
        mid-write never leaves a truncated index."""
        directory = self.endpoint_dir(endpoint)
        tmp = directory / "manifest.json.tmp"
        with open(tmp, "w") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp, directory / "manifest.json")

    @staticmethod
    def _fill_empty(df: pd.DataFrame) -> pd.DataFrame:
        for col, default in _default_row().items():
            if col not in df.columns:
                continue
            if col == "warnings":
                df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
            else:
                df[col] = df[col].fillna(default)
        df["flexibility_success"] = df["flexibility_success"].astype(bool)
        return df

    def _create_output(self, data: pd.DataFrame) -> ActorOutput:
        succ = data[data["flexibility_success"]] if "flexibility_success" in data else data.iloc[:0]
        return ActorOutput(
            data=data, success=True,
            metadata={"n_molecules": len(data), "n_success": len(succ),
                      "mean_V": float(succ["mean_V"].mean()) if len(succ) else 0.0,
                      "endpoints": list(ENDPOINTS)},
            endpoint=self.forge_endpoint,
        )
