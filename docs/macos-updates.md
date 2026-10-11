# macOS releases and updates

## Current distribution boundary (2026-10-11)

GitHub `v0.4.0` has no uploaded assets: it is a source release. The app is currently built with ad-hoc signing, Hardened Runtime enabled, arm64, macOS 15.0 minimum, marketing version `0.4.0`, and build `1`. The inspected local Keychain offers Apple Development identities, but no Developer ID Application identity. An Apple Development certificate does not satisfy public Developer ID distribution.

The app checks at startup and then at most once per day, without a startup wait or recurring dialogs. Settings can disable automatic checks. Successful results and ETags are cached for up to seven days, with a timestamp shown; network failures back off automatic checks for an hour, and GitHub rate limits also delay manual checks. A manual check can retry other failures immediately.

The app checks the official repository's GitHub releases and offers **View Release**. It does not download or replace the app, embed Sparkle, upgrade Python, run `pip`/`pipx`, or modify the original engine environment. The user manages that environment independently. The HTTP engine contract uses `api_version: 1`; `engine_version` identifies the engine package separately from the app version.

Future releases must include this compatibility marker in their GitHub release body:

```html
<!-- iris-macos: {"minimumSystemVersion":"15.0","architectures":["arm64"]} -->
```

Use the actual minimum OS and supported architectures for each release. A future release without valid compatibility metadata is not automatically announced as compatible. The current app `0.4.0` baseline is known; this does not imply that future releases share its requirements. Drafts and prereleases are excluded. An unavailable network or malformed response must preserve the installed app and saved data.

## Python engine

The app supports engine `0.3.x` with HTTP protocol `api_version: 1`. The current source resolves to engine `0.3.0`. The released `v0.3.0` server predates both health version fields: it can attach with a notice that compatibility cannot be fully verified. An explicitly incompatible engine version or API version is rejected. App updates do not update the engine.

In Settings, record the current engine executable path before changing it. For pipx installations, the usual launcher is `~/.local/bin/iris-metal`; use `pipx list` to identify its managed environment, since its location can vary. A virtual-environment installation instead uses that environment's `bin/iris-metal`; use its own `bin/python -m pip`, rather than an unrelated global `pip`. Keep the existing environment, its install source/version, and the executable setting available for recovery.

For a bounded manual change, finish or cancel generation, quit Iris, and stop any separately started server through the terminal/session that owns it. Create a **new** candidate virtual environment from an approved, pinned official release; retain the old pipx or virtual environment. These example commands target the existing Python engine `v0.3.0` baseline and require Python 3.11 or newer, not an unreviewed future release. They download package dependencies, but do not download model weights:

```sh
iris_candidate="$HOME/.local/share/iris-engine-candidate-v0.3.0"
test ! -e "$iris_candidate" && \
  python3 -m venv "$iris_candidate" && \
  "$iris_candidate/bin/python" -m pip install "git+https://github.com/TKY-27/iris-3b-metal@v0.3.0" && \
  "$iris_candidate/bin/iris-metal" --help
```

If the directory exists, choose another new path; do not remove or overwrite it. After a successful installation, select the candidate's absolute `bin/iris-metal` path in Iris Settings and reconnect. Confirm engine health and a small generation using the existing local checkpoint before adopting it. On failure, stop the candidate server, restore the original executable path, and reconnect to the retained environment. Preserve chats, images, settings, and checkpoints throughout. Do not run arbitrary remote installation scripts or upgrade the working environment in place as part of the GUI update flow.

## Preparing a distributable archive

`apps/macos/package_release.sh` provides a bounded archive/export workflow. No release is uploaded by it. Its default use here is input validation; `--execute` deliberately performs an Apple notarization submission and must only be used after that operation is authorized.

Missing prerequisites are a valid **Developer ID Application** certificate and private key for the intended Apple Developer team, an existing `notarytool` Keychain credential profile, and an approved release version/build. Neither the script nor this document creates credentials or signing keys. A full Xcode installation is required. Choose a new absolute output directory; existing output is refused. Builds must increase monotonically across releases, including patch releases; the script validates the numeric format, but the release owner must compare against published history.

Set these environment values to your real release settings locally; do not commit them or credential material:

| Variable | Required value |
| --- | --- |
| `IRIS_RELEASE_VERSION` | Three numeric components, e.g. current app version `0.4.0` |
| `IRIS_RELEASE_BUILD` | Integer 1–9999, greater than all previously published builds |
| `IRIS_DEVELOPER_ID` | Full existing `Developer ID Application: … (TEAMID)` identity name |
| `IRIS_TEAM_ID` | Actual 10-character team identifier |
| `IRIS_NOTARY_PROFILE` | Name of an existing Keychain profile; never a password |
| `IRIS_RELEASE_OUTPUT` | New absolute directory for this run |

Validate inputs without creating files, reading credentials, building, or accessing the network:

```sh
apps/macos/package_release.sh --dry-run
```

Once signing and Apple submission are authorized, `--execute` archives Release with explicit signing/version settings, exports with the Developer ID method, and checks bundle identity, versions, architecture, minimum OS, signature, team, and Hardened Runtime. It then submits a ZIP with `notarytool --wait`, requires `Accepted`, staples and validates the app, applies Gatekeeper assessment, and creates a fresh final ZIP plus SHA-256 file. The final archive contains only the app; the submission ZIP predates stapling and is not the distribution artifact. The output also retains archive/export/notary results for diagnosis. A failed run leaves its output for review; use a new directory for another attempt.

This workflow has only been syntax- and dry-run-tested without distribution credentials. It is not evidence of a successfully signed, notarized, installed, or updateable release. Before publishing, inspect the artifact on a clean supported Mac, launch it through Gatekeeper, and confirm the app still uses the user's configured engine and preserves the library. Publishing binaries, modifying a release, or hosting an update feed requires separate authorization.

## Future Sparkle 2 rollout

Add Sparkle only when signed binary distribution and its maintenance are established. Follow [Sparkle's official setup](https://sparkle-project.org/documentation/) and [customization reference](https://sparkle-project.org/documentation/customization/): embed and sign its framework/helpers, use an official HTTPS appcast, embed the EdDSA public key in `SUPublicEDKey`, and keep the private key in the release owner's Keychain. Never commit the private key. Sign the final stapled archive with Sparkle's tools and enable `SUVerifyUpdateBeforeExtraction`; app code signing and archive authentication protect different stages. Feed metadata must include the real minimum OS and an increasing `CFBundleVersion`. Pin and review the chosen Sparkle 2 version before integration.

After the signing identity, hosting location, and release are approved, the release owner performs key generation once using the chosen Sparkle distribution's tools. These commands are documented steps only; they have not been executed:

```sh
# Set SPARKLE_BIN to the reviewed Sparkle distribution's bin directory.
"$SPARKLE_BIN/generate_keys"
# Set IRIS_UPDATE_ARCHIVES to a local directory containing validated final archives.
"$SPARKLE_BIN/generate_appcast" "$IRIS_UPDATE_ARCHIVES"
```

[Sparkle’s official publishing guide](https://sparkle-project.org/documentation/publishing/) describes archive signatures and appcast fields. `generate_keys` stores the private key in the login Keychain and returns the public key for the app. `generate_appcast` signs archives and produces appcast metadata; review the generated versions, signatures, lengths, OS requirements, and URLs before use. Publish validated archives to an approved GitHub release manually, then set enclosure URLs to the corresponding official HTTPS assets under `https://github.com/TKY-27/iris-3b-metal/releases/download/<approved-tag>/<validated-archive-name>`. The angle-bracket components are placeholders, not deployed URLs. Configure the separately approved official HTTPS appcast URL in `SUFeedURL`; no feed endpoint is currently deployed by this work. Re-sign changed feed/metadata as required by the selected Sparkle configuration, and publish the reviewed feed only after the referenced artifacts are available and validated.

Iris-specific replacement must wait until generation is idle, or explicitly cancel and wait for the active job to stop. Block new generation, shut down the owned engine process safely, flush pending library writes, and then let Sparkle replace and restart the app. Preserve externally managed engine processes/environments, chat/library data, images, checkpoints, and user settings. Do not replace a Python environment as part of an app update. Resume normal operation only after the restarted app has checked the engine protocol compatibility.

Before claiming automatic updates work, test an actual old signed/notarized app to a newer signed/notarized app end to end, including restart and library persistence. Exercise idle and active generation, cancellation, pending library edits, missing/incompatible engine, unsupported OS/architecture, offline and interrupted download, altered archive/signature, failed installation, and read-only/translocated installation. Failures must preserve the old working app and user data. Record the artifacts, versions, signing/notarization evidence, and observed results. A mocked feed, compilation, or dry-run alone cannot establish this behavior.
