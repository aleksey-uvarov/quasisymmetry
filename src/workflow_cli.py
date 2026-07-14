"""Shared CLI vocabulary for reference state vs computational backend.

Across the pipeline the two axes are:

* ``--reference`` — which state/energy is the ground-truth reference
  (``fci`` / ``hf`` / ``dmrg``). Used by orbital optimization and by metrics
  for ``dE`` / reference-ordered ``K``.
* ``--backend`` — how expensive linear-algebra is done
  - optimize: ``statevector`` (ffsim/FCI) or ``dmrg`` (Block2 MPS costs)
  - metrics: ``fci`` (eigsh/eigh sectors), ``davidson`` (PySCF Davidson),
    or ``dmrg`` (Block2 sector GS)

``dmrg`` is the same Block2 stack wherever it appears; common flags
(``--bond_dim``, ``--wavefunction_dir``, ``--n_threads``, …) are shared.
"""

from __future__ import annotations

import argparse

REFERENCE_CHOICES = ("fci", "hf", "dmrg")
OPTIMIZE_BACKEND_CHOICES = ("statevector", "dmrg")
METRICS_BACKEND_CHOICES = ("fci", "dmrg", "davidson")


def add_dmrg_common_args(parser: argparse.ArgumentParser) -> None:
    """Bond dimension / store / threads used by any ``dmrg`` path."""
    parser.add_argument(
        "--bond_dim",
        type=int,
        default=250,
        help="DMRG bond dimension (--reference/--backend dmrg)",
    )
    parser.add_argument(
        "--wavefunction_dir",
        default=None,
        help="local DMRG wavefunction store to reuse/create",
    )
    parser.add_argument(
        "--n_threads",
        type=int,
        default=4,
        help="block2 threads (dmrg reference / backend)",
    )


def add_optimize_workflow_args(parser: argparse.ArgumentParser) -> None:
    """``--reference`` / ``--backend`` for ``optimize_symmetries.py``."""
    parser.add_argument(
        "--reference",
        choices=REFERENCE_CHOICES,
        default="fci",
        help="reference state: fci (default), hf, or dmrg (Block2 MPS; "
             "with --backend statevector the MPS is converted to a CI vector)",
    )
    parser.add_argument(
        "--backend",
        choices=OPTIMIZE_BACKEND_CHOICES,
        default="statevector",
        help="cost backend: statevector (ffsim/FCI, default) or dmrg "
             "(MPS-native NC/variance on a fixed DMRG reference)",
    )
    add_dmrg_common_args(parser)


def add_metrics_workflow_args(parser: argparse.ArgumentParser) -> None:
    """``--reference`` / ``--backend`` for ``metrics.py`` sector diagnostics."""
    parser.add_argument(
        "--backend",
        "--solver",
        dest="backend",
        choices=METRICS_BACKEND_CHOICES,
        default="fci",
        help="sector eigensolver backend: fci (eigsh/eigh, default), "
             "davidson (PySCF Davidson on the same sector blocks), or "
             "dmrg (Block2 MPS-native E_dec/K). --solver is an alias.",
    )
    parser.add_argument(
        "--reference",
        choices=("fci", "dmrg"),
        default=None,
        help="reference energy/state for dE and reference-K "
             "(default: dmrg when --backend dmrg, else fci)",
    )
    add_dmrg_common_args(parser)


def resolve_metrics_reference(backend: str, reference: str | None) -> str:
    """Pick metrics reference; DMRG backend defaults to a DMRG reference."""
    if reference is not None:
        return reference
    return "dmrg" if backend == "dmrg" else "fci"
