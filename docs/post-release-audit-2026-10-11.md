# Post-release audit — 2026-10-11

Baseline: `0beea7be41b9b42a592c08f7411d5d38930c4d5f` on
`codex/final-independent-audit`. The incoming benchmark script/test changes,
report, and untracked shadow diagnostic were saved before edits in
`.scratch/post-release-audit/`. No release, push, remote setting change,
checkpoint download, or inference optimization was performed.

## Findings and changes

- `qk_norm_rope` accepted negative offsets before Metal pointer arithmetic.
  Reject negative offsets and nonpositive head counts before dispatch. Valid
  numerical paths, dtypes, sampling defaults and kernels remain unchanged.
- Interrupted legacy library migration could delete the only image copy for
  an already-indexed entry. Copy missing images before removing History;
  suppress duplicate legacy IDs in the same migration.
- Read-only libraries allowed internal entry/outcome/image mutations. Those
  entry points now refuse writes.
- Cancellation during result download could publish a completed image.
  Check cancellation around async boundaries and wait for outcome persistence
  before quit proceeds.
- Engine health now reports package and API versions. The GUI distinguishes
  them from its own version, rejects explicitly incompatible engines, and
  identifies legacy servers whose compatibility cannot be fully verified.
- Add native stable-release detection, manual/app-menu and daily automatic
  checks, ETag caching, rate-limit handling, stale-result expiry, plain release
  information, and View Release. Compatibility metadata is required for
  future release notifications. No code or update artifacts are downloaded.
- Remove the abandoned fresh-shadow implementation, associated CLI/test
  machinery, and experimental CFG-interval flag. Preserve the original
  diagnostic in the ignored audit archive and keep its benchmark evidence.
  Retain batch fixture support, optional prompt profiling, and branch-order
  regression coverage. There were no latest-experiment runtime toggles in
  `src` to remove. Existing Qwen streaming/adapted caches remain in use.

Review covered kernels/precision and MPS ownership, text/cache/loading,
sampling/cancellation, HTTP boundaries, CLI, Swift process state, library
migration/persistence, and image workflows. Existing regression coverage was
used for unchanged paths; this was not another performance investigation.

## Verification

| Check | Result |
| --- | --- |
| Offline full Python suite | 403 passed; six existing MPS pin-memory warnings |
| Focused Python/MPS suite | 177 passed |
| Tiny actual MPS batch fixture | n=2, 24×16, two steps; finite output |
| Swift suite with CPU stand-in HTTP engine | 58 passed, zero skipped |
| Final update parser/cache regression suite | 17 passed |
| macOS Debug and Release builds | Passed, local ad-hoc signing |
| Ruff lint/format and Git whitespace | Passed |
| Offline wheel and source archive | Built; wheel excludes weights/local caches |
| Packaging script | Shell syntax, dry-run inputs, missing-identity refusal passed |
| Signed/notarized packaging and real self-install | Not run; prerequisites unavailable |
| Full checkpoint generation/benchmark | Not run; no changed inference arithmetic |

The native app menu successfully queried the real official GitHub API and
showed no newer compatible release. A local cached fixture exercised the
update toolbar indicator, release details, settings navigation, and return to
no-update state. A CPU stand-in exercised active batch generation while the
update window remained informational, followed by cancellation and accessible
multiple-image results. These images were synthetic, not quality evidence.

GUI runs used isolated test libraries. A tool observation briefly relaunched
without the test environment; that instance was immediately quit. Before/after
checks found no content or mtime changes in the 344 protected files (existing
library/images/release assets checked by hash, large checkpoint files by
size/inode/mtime). Test preference fixtures were restored. No destructive
operations targeted those files.

Rebuildable Python/Xcode caches were removed. The recorded first cleanup
removed 261,558,272 allocated bytes (249.4 MiB); filesystem free space rose
255,741,952 bytes at that observation. This is cache reclamation, not a net
repository-size or physical APFS exclusivity claim. Later verification rebuilt
some caches; those were removed again without double-counting the savings.
Original experiments, generated images, traces, benchmark data, and useful
release/build products remain. Logs/manifests are in
`.scratch/post-release-audit/` and are not distribution assets.

## Distribution boundary

GitHub `v0.3.0` has no attached application assets. The current build is
ad-hoc signed, and no Developer ID Application identity was available.
**One-click installation is not implemented or claimed.** The new packaging
script and [update documentation](macos-updates.md) prepare the next steps:
Developer ID signing, notarization, an official signed Sparkle appcast and
archive-verification key, then a real old-to-new upgrade test. Keys stay out
of Git. Publication still requires explicit authorization.

The UI follows macOS settings/menu/window conventions, with no repeated
notification dialogs or decorative card system. References reviewed:
[Apple Software Update](https://support.apple.com/guide/mac-help/get-macos-updates-mchlpx1065/mac),
[Impeccable's slop catalog](https://impeccable.style/slop/),
[additional design-pattern critique](https://smoothui.dev/blog/ai-design-slop),
and [Sparkle's official setup](https://sparkle-project.org/documentation/).
