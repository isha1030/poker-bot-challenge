# Strategy validation

Run `python -m pip install -r requirements-dev.txt`, then `python -m pytest -q`.

The strategy uses sampled short-deck Omaha equity with exact showdown evaluation,
including split pots and excluding all known cards. It uses pot odds, modest
pressure adjustments, bounded sampling, and per-hand caching. These are practical
heuristics, not equilibrium play or a fully inferred opponent range.

Safeguards: validate cards, check returned action legality, use check-or-fold on
internal failures, keep bounded error diagnostics, clear memory every half, and
cache repeated board evaluations. `memory['errors']` must remain zero in matches.
The compute quota is adaptive and reserves 65% of the configured bank for other
work; eight minimum samples and scheduling overhead mean it is not a hard timer.
Always verify the real judge's timing on the deployment machine.

Tests cover the original HTTP contract, actual seat swapping, impossible straight
bonuses, four hole aces, fault injection, illegal policy outputs, duplicate cards,
price-sensitive calling, cached estimates, and memory reset.

To reproduce adversarial matches against an earlier agent saved outside this repo:

```
python benchmark_strategy.py path/to/previous_agent.py results.json
```

This runs 1,000 paired deals per seed, both seats, against the previous policy,
always-pot, always-call, and a 12-sample equity challenger. The challenger uses the
previous agent's preflop logic. No opponent receives hidden cards. Timing can affect
adaptive sample counts, so outcomes need not be bit-for-bit identical across hosts.

Initial validation (three holdout seeds, 2,000 hands per match):
- Previous policy: +6.061, +6.399, +6.624 bb per paired deal; all 95% intervals positive.
- Always-pot: +5.487, +6.452, +5.471; all intervals positive.
- Always-call: +6.552, +6.414, +6.605; all intervals positive.
- Equity challenger: -0.187, -0.907, -0.894; each interval includes zero.
- Zero internal strategy errors across all 24,000 hands.

The equity challenger remains a meaningful weakness: all three point estimates
were negative. Do not claim this policy is unbeatable. Improvements should be
accepted against multiple holdout opponents and full HTTP timing, not one seed.

Full HTTP judge validation: 2,000 hands, seed http-holdout-101, +6.835 bb/deal (95% CI +5.360 to +8.310), zero errors, charged time 3.73s and 4.18s of 8s. Use explicit http://127.0.0.1:PORT on Windows if localhost health checks time out.

