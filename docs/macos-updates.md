# macOS releases and updates

## Current distribution

The v0.5.0 distribution is a [GitHub source release](https://github.com/TKY-27/iris-3b-metal/releases/tag/v0.5.0). It contains source code only: there is no signed or notarized app download and no model checkpoint attached. Build the app from source using the instructions in the [README](../README.md#macos-app).

The macOS app and Python engine are version `0.5.0`; the app build is `1`. The app supports arm64 and macOS 15.0 or later. HTTP API version remains `1`. The app supports engines `0.3.x` and `0.5.x` with API `1`; Turbo requires the engine to advertise Turbo support.

## App update checks

The app checks the official GitHub Releases API and can show compatible stable releases and their notes. **View Release** opens the release page. The app does not download, replace or install itself, and it does not update Python or change the configured engine environment. Automatic checks can be disabled in Settings. When enabled, release checks contact GitHub.

Future releases must include a compatibility marker in the release description. Use the actual minimum macOS version and architectures for that release:

```html
<!-- iris-macos: {"minimumSystemVersion":"15.0","architectures":["arm64"]} -->
```

Drafts, prereleases and releases without valid compatibility metadata are not offered as compatible app releases.

## Updating the Python engine

The app uses a separately installed `iris-metal` executable and never changes its Python environment. To try the engine packaged in repository release `v0.5.0` without replacing an existing installation, create a new virtual environment:

```sh
candidate="$HOME/.local/share/iris-engine-v0.5.0"
test ! -e "$candidate" && \
  python3 -m venv "$candidate" && \
  "$candidate/bin/python" -m pip install "git+https://github.com/TKY-27/iris-3b-metal@v0.5.0" && \
  "$candidate/bin/iris-metal" --help
```

If the candidate directory already exists, choose another new path; do not replace an existing environment. This installs Python engine `0.5.0`. The older repository tag `v0.3.0` predates engine version fields, so the app can connect to it only with a notice that compatibility cannot be fully verified. The current `v0.5.0` source reports engine and API versions.

After confirming the candidate command works, select its absolute `bin/iris-metal` path in Iris Settings and check the engine status. Keep the previous environment until a small generation succeeds. This install command downloads Python package dependencies; model weights download separately on first use. The app never modifies the image library, images or model checkpoints during an engine change.

## Maintainer packaging

[`apps/macos/package_release.sh`](../apps/macos/package_release.sh) can build, sign, notarize and package a local app archive. `--dry-run` validates inputs only; `--execute` submits the app to Apple and never uploads a GitHub release. Execution requires an existing Developer ID Application identity, a Keychain notary profile, an approved version/build and a new output directory. The current `v0.5.0` release has no signed or notarized app archive.
