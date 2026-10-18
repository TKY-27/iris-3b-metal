# macOS releases and updates

## Install the macOS release

The v0.6.0 release contains a native arm64 app ZIP and a matching engine wheel:

- `Iris-0.6.0-1-arm64.zip`
- `iris_3b_metal-0.6.0-py3-none-any.whl`
- `iris-engine-0.6.0-constraints.txt`
- `SHA256SUMS`

Download all four files from the [v0.6.0 GitHub release](https://github.com/TKY-27/iris-3b-metal/releases/tag/v0.6.0). In Downloads, run `shasum -a 256 -c SHA256SUMS` to check the ZIP, wheel and constraints file before installing. The ZIP contains `Iris.app`. The app is ad-hoc signed, not signed with a Developer ID, and not notarized. The wheel contains the Python engine package, not Python itself or model weights.

Install [Python 3.14](https://www.python.org/downloads/macos/) first. Python 3.14.8 is the tested version. Create a new virtual environment at its final location, then install the downloaded wheel with the matching tested dependency constraints:

```sh
python3.14 --version
engine="$HOME/.local/share/iris-engine-0.6.0"
if [ -e "$engine" ] || [ -L "$engine" ]; then
  printf 'Choose a new environment path: %s already exists\n' "$engine" >&2
  exit 1
fi
python3.14 -m venv "$engine"
"$engine/bin/python" -m pip install \
  -c "$HOME/Downloads/iris-engine-0.6.0-constraints.txt" \
  "$HOME/Downloads/iris_3b_metal-0.6.0-py3-none-any.whl"
"$engine/bin/iris-metal" --help
```

Download the wheel and `iris-engine-0.6.0-constraints.txt` into `Downloads` before running these commands. If the target directory already exists, choose another new path. In Iris Settings, set the engine path to the absolute path `$HOME/.local/share/iris-engine-0.6.0/bin/iris-metal`, replacing `$HOME` with your home directory. The app launches that executable as a separate process and does not change the Python environment. Keep the previous engine environment until a small generation succeeds.

Unzip `Iris-0.6.0-1-arm64.zip`, then drag `Iris.app` into Applications. Moving the app does not move or change the Python environment or the image library at `~/Library/Application Support/Iris`. Keep the engine at its installed path.

The wheel declares Python dependencies that `pip` installs from the package index. The install needs network access for dependencies. The first generation downloads the selected text-to-image checkpoint and the shared Qwen3-VL-4B-Instruct text encoder from Hugging Face. Original weights are about 12.0 GB, Turbo weights are about 6.0 GB, and the text encoder is about 8.9 GB. Depth and restoration weights, about 12 GB each, download when first used. The release contains no model weights. Allow enough disk space for downloads, caches, and generated images.

The app requires macOS 15 or later. Release runtime testing covered an M5 Mac with 24 GB of memory. Runtime behavior on macOS 15 and on 16 GB Macs has not been tested.

### Approve the first launch

Because the app is not notarized, macOS may block its first launch. On macOS 27, open **System Settings > Privacy & Security > Security**, choose **Open** beside the Iris notice, then choose **Open Anyway** and authenticate if macOS asks. Apple's [macOS 27 guide](https://support.apple.com/guide/mac-help/open-an-app-from-an-unknown-developer-mh40616/mac) documents this order. Other macOS versions may use different labels or order. Apple's [safely opening apps](https://support.apple.com/en-us/102445) page describes **Open Anyway**, then **Open**.

Approve Iris only if you downloaded it from the official release and verified its checksum. Do not disable Gatekeeper globally, remove quarantine attributes, or use Control-click as a workaround. This approval only addresses Gatekeeper's first-launch block. It does not repair a damaged signature, an incomplete download, or missing runtime dependencies.

## App update checks

The app checks the official GitHub Releases API and can show compatible stable releases and their notes. **View Release** opens the release page. The app does not download, replace, or install itself, and it does not update Python or change the configured engine environment. Automatic checks can be disabled in Settings. When enabled, release checks contact GitHub.

Release descriptions must include a compatibility marker with the minimum macOS version and supported architectures:

```html
<!-- iris-macos: {"minimumSystemVersion":"15.0","architectures":["arm64"]} -->
```

Drafts, prereleases, and releases without valid compatibility metadata are not offered as compatible app releases.

## Engine compatibility and updates

The v0.6.0 app uses HTTP API version 1 and accepts engines 0.3.x, 0.5.x, and 0.6.x that report API version 1. Turbo also requires the engine to advertise Turbo support. The app runs the selected `iris-metal` executable on localhost. It does not bundle Python or change the selected environment.

To update the engine, download the wheel from the same release as the app and install it into a new virtual environment using the steps above. Do not replace an environment that another app or task uses. The engine can also be installed separately for CLI or Python use.

## Build the app from source

If you want to build the app yourself, install Xcode 27 and follow the [source build instructions in the README](../README.md#macos-app). The downloaded app route above does not require Git or Xcode.

## Maintainer packaging

`apps/macos/package_release.sh` supports an ad-hoc mode for local ZIP creation and a separate Developer ID mode for notarized packaging. Ad-hoc mode needs macOS and Xcode, but does not inspect Apple credentials or contact Apple. Dry-run validates only the supplied values and creates no files:

```sh
IRIS_RELEASE_VERSION=0.6.0 \
IRIS_RELEASE_BUILD=1 \
IRIS_RELEASE_OUTPUT=/absolute/path/to/new-output \
  apps/macos/package_release.sh --mode ad-hoc --dry-run
```

Run the same command with `--execute` to build the arm64 app ZIP. The output path must not exist. This script does not build the Python wheel, package the constraints file, create `SHA256SUMS`, or publish a release. Build and verify the wheel separately, then confirm its version matches the app before attaching both artifacts to a release. The script's Developer ID mode remains available for a notarized distribution.
