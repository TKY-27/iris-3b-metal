Paste the block below after `/goal` in a Claude Code session started in this
checkout on the `opt/loop` branch, with ultracode effort and auto mode.

```
Optimize Iris-3B Metal inference, Original and Turbo, by following CLAUDE.md exactly, until all of these appear in this conversation: (1) the output of `python loop/gate.py stopcheck` ending in `GATE STOPCHECK MET`; (2) after it, a fresh loop-auditor subagent whose result ends in `AUDIT: AGREE`; (3) the output of `python loop/gate.py audit` ending in `GATE AUDIT written to`; (4) the URL of a pull request from opt/loop to main with the final report in benchmarks/. Every merged change must show its `GATE Gxxxx VERDICT=ACCEPT` line, a code-reviewer `VERDICT: PASS`, and a `merged Gxxxx` line. Never edit the frozen harness. Stop and wait for the person, without clearing this goal, if a `GATE ERROR` reports a modified harness, if three verdicts in a row are NOISY, or if the Mac needs physical attention.
```
