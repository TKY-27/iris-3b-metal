Paste the block below after `/goal` in a Claude Code session started in this
checkout on the `opt/loop` branch, with ultracode effort and auto mode.

```
Optimize Iris-3B Metal inference, Original and Turbo, by following CLAUDE.md exactly. The goal is met when this conversation shows, in this order: (1) a `GATE STOPCHECK MET` line with no `GATE merged` line after it; (2) after it, a fresh loop-auditor subagent result whose last line is `AUDIT: AGREE`; (3) a `GATE AUDIT PASS` line, which itself verifies every merge, review, the stopcheck and the branch history; (4) a `GATE FINAL READY` line; (5) the URL of a pull request from opt/loop to main whose description links the final report in benchmarks/. The goal is also over, and nothing more should be attempted, once the conversation shows a line starting `LOOP HALTED:` written for one of the reasons in CLAUDE.md (a HALT error, three NOISY or three ERROR verdicts in a row, or three auditor disagreements). Never edit the frozen harness and never work around a verdict.
```
