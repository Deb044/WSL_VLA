# Methodology 1 Implementation Status

“Implemented” means code and host contract tests exist; it does not mean the
official WSL/CUDA experiment has already been executed.

| Plan area | Status | Primary implementation |
|---|---|---|
| Pinned Octo-Small and LIBERO provenance | Implemented | `wsl_vla/octo/bridge.py`, `scripts/research_preflight.py` |
| Full installed-environment lock and input hashes | Implemented | `scripts/research_preflight.py` |
| Real episode-level dataset splits | Implemented | `wsl_vla/data/libero.py`, `wsl_vla/data/splits.py` |
| Modality adapters and diffusion-head updates | Implemented | `wsl_vla/adapters/flax.py`, `wsl_vla/octo/bridge.py` |
| Basis-invariant effective-update packing | Implemented | `wsl_vla/adapters/packing.py` |
| 40×3×3 model zoo contracts | Implemented | `scripts/train_research_zoo.py`, `scripts/build_alignment_archive.py` |
| Research-zoo integrity and completeness gate | Implemented | `scripts/verify_research_zoo.py` |
| Visual/language/action evidence | Implemented | `scripts/extract_research_evidence.py`, `wsl_vla/alignment/octo_evidence.py` |
| Joint alignment and reconstruction control | Implemented | `scripts/train_research_alignment.py` |
| Linear mapper and empirical token shells | Implemented | `wsl_vla/alignment/mapping.py`, `wsl_vla/alignment/checkpoint.py` |
| Held-out alignment advantage gate | Implemented | `scripts/run_alignment_advantage_gate.py` |
| Differentiable JAX latent refinement | Implemented | `wsl_vla/alignment/models.py`, `wsl_vla/adapters/latent.py` |
| Validation-frozen gamma/patience selection | Implemented | `configs/research/gamma_candidates.yaml`, `scripts/run_gamma_validation.py` |
| OOD five-method protocol | Implemented | `scripts/run_official_ood.py` |
| Four-fold OOD reporting | Implemented | `scripts/report_ood_study.py` |
| Shared-state continual protocol and eight controls | Implemented | `scripts/run_official_continual.py` |
| One-task and two-task scaling gates | Implemented | `scripts/run_one_task_learning_gate.py`, `--task-count 2` |
| One-suite/three-seed/eight-condition pilot gate | Implemented in job graph | `scripts/plan_methodology1_jobs.py` |
| Consecutive component swaps | Implemented | `scripts/run_component_swaps.py` |
| Component-swap aggregation and drift correlations | Implemented | `scripts/report_component_swaps.py` |
| Recovery-step curves | Implemented for LoRA shared runs | `scripts/run_recovery_probe.py` |
| Four-fold/three-seed reporting | Implemented | `scripts/report_publication_study.py` |
| Machine-checkable completion/provenance audit | Implemented | `scripts/audit_methodology1_study.py` |
| Distributed job graph and execution ordering | Implemented | `scripts/plan_methodology1_jobs.py` |
| Official WSL/CUDA integration execution | Not yet evidenced | Run `METHODOLOGY1_RUNBOOK.md` |
| Final research numbers and figures | Not yet produced | Requires collaborator compute |
| Large-VLA confirmation | Explicitly deferred | Cluster phase after Octo-Small |

Open design and feasibility concerns are tracked in
`docs/IMPLEMENTATION_REVIEW_2026-10-02.md`.

The Windows suite verifies schemas, metrics, leakage checks, and state
semantics. Publication claims begin only after official WSL preflight,
zero-equivalence, simulator gates, and rollout experiments succeed.
