## Lean Reboot Inventory

| Category | Keep | Rationale | Action |
| --- | --- | --- | --- |
| Core C++ engine | `cpp/`, `include/`, top-level `CMakeLists.txt`, `pyproject.toml` | Implements pybind11 extension, state logic, and build scripts covered by `TEST_ARCHITECTURE.md`. | **Keep** |
| Build outputs | `build/`, `$PWD/ofc_cpp.cp312-win_amd64.pyd` | Required for Windows builds but can be regenerated. | Keep for now; document regeneration steps. |
| Python env/bridge | `ofc_env.py`, `state_encoding.py`, `policy_net.py`, `policy_net_policy.py`, `train_policy_net.py`, `collect_sfl_dataset_cpp.py`, `policy_net_epoch*.pth`, `policy_net_sfl_cpp*.pth`, `data/sfl_cpp_dataset/` | Minimal pipeline: C++ -> Python batching -> CUDA training/eval. | **Keep** |
| Architecture docs | `TEST_ARCHITECTURE.md`, `ARCHITECTURE_SUMMARY.md`, `WINDOWS_SETUP.md`, `LINUX_SETUP.md`, `SETUP_INSTRUCTIONS.md`, `TRAINING_FLOW.md` | Explain the retained pipeline; keep for onboarding. | **Keep** |
| Legacy RL / ValueNet stack | `train.py`, `simple_train.py`, `supervised_train.py`, `action_head_train.py`, `heuristic_imitation_train.py`, `two_player_sim.py`, `single_player_eval.py`, `debug_action_preferences.py`, `collect_heuristic_dataset.py`, `evaluate_*`, `plan_cpp_*`, `Optimization_Summary.md`, `targets.py`, `ValueNet` checkpoints, `policy_net_policy.py`? (keep) | Obsolete after reboot; relied on ValueNet trunk + action head and Python rollouts. | **Remove/Archive** |
| Legacy policies + scripts | `heuristic_policy.py`, `sfl_policy.py` (Python), `ismcts.py`, `demo.py`, `test_*` scripts tied to ValueNet, `benchmarks/` | Will be superseded by C++ bots from `xeond8` repo; currently unused. | **Remove/Archive** once new C++ equivalents are in place. |
| Old datasets/checkpoints | `data/heuristic_dataset/`, `value_net_checkpoint_*.pth`, `supervised_value_net.pth`, `value_net.pth` | Specific to retired architectures. | **Delete** after backup if needed. |
| Misc setup fixes | `FIX_*`, `ACTIVATE_ENV.md`, `fix_pybind11.sh`, `plan_cpp_fast_dataset.md` etc. | Some redundant; evaluate per file. | Move essential notes into README; delete rest. |

**Next step:** confirm this keep/remove matrix, then delete/archive the marked files and prune `requirements.txt` accordingly. Let me know if any items should be moved between columns before I start deleting.***

