## Native SFL Port Notes

This repo now mirrors the **Simple Fantasy Like** (SFL) agent described in the [`xeond8/OFC-Poker-Agents` bot](https://github.com/xeond8/OFC-Poker-Agents/tree/master/bot)'s `SimpleFantasyLikeOneBoard` implementation:

- **Monte Carlo completions.** Every legal move is evaluated by sampling future decks, exhaustively completing the remaining board slots, and scoring the best royalty outcome for that sample. This replaces the earlier heuristic row-strength proxy.
- **Full hand validation.** Completed boards are ranked with the same royalty and foul rules used elsewhere in the engine (`Rules.hpp`), so SFL never “likes” an ordering that would be dead in real play.
- **Deterministic sampling.** Samples are derived from a per-state FNV hash plus a xorshift mix. Dataset collection and unit tests now get reproducible targets while still emulating the stochastic completions from the reference agent.
- **C++-first data flow.** The pybind helpers (`sfl_choose_action`, `generate_sfl_dataset`) now reuse this Monte Carlo scorer, ensuring that imitation data, online inference, and future search policies all agree on the same action quality signal.

See `cpp/sfl_policy.cpp` for the C++ port and `tests/test_sfl_policy.py` for parity checks against a pure-Python translation of the same logic.

