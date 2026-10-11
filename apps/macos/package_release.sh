#!/bin/bash
# A new local output directory only. No publishing; --execute submits to Apple.
set -euo pipefail

fail() { printf 'Error: %s\n' "$1" >&2; exit 1; }
usage() {
    cat <<'USAGE'
Usage: package_release.sh --dry-run | --execute
Required environment: IRIS_RELEASE_VERSION, IRIS_RELEASE_BUILD,
  IRIS_DEVELOPER_ID (full Developer ID Application identity name),
  IRIS_TEAM_ID, IRIS_NOTARY_PROFILE (existing Keychain profile), IRIS_RELEASE_OUTPUT.
--dry-run validates inputs without building, signing, network access, or creating files.
--execute archives, exports, submits to Apple for notarization, staples, and packages.
It never uploads a release. The output directory must not already exist.
USAGE
}
[[ $# == 1 ]] || { usage; exit 1; }
case "$1" in
    --help) usage; exit 0 ;;
    --dry-run) execute=false ;;
    --execute) execute=true ;;
    *) usage; exit 1 ;;
esac
for key in IRIS_RELEASE_VERSION IRIS_RELEASE_BUILD IRIS_DEVELOPER_ID IRIS_TEAM_ID IRIS_NOTARY_PROFILE IRIS_RELEASE_OUTPUT; do
    [[ -n "${!key:-}" ]] || fail "Missing $key"
done
[[ "$IRIS_RELEASE_VERSION" =~ ^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]] || fail 'Version must be major.minor.patch'
[[ "$IRIS_RELEASE_BUILD" =~ ^[1-9][0-9]{0,3}$ ]] || fail 'Build must be 1–9999 and higher than every published build'
[[ "$IRIS_TEAM_ID" =~ ^[A-Z0-9]{10}$ ]] || fail 'Team ID must have 10 uppercase letters/digits'
[[ "$IRIS_DEVELOPER_ID" == 'Developer ID Application: '*" ($IRIS_TEAM_ID)" ]] || fail 'Developer ID Application identity must match the team'
[[ "$IRIS_NOTARY_PROFILE" != *$'\n'* && "$IRIS_NOTARY_PROFILE" != *$'\r'* ]] || fail 'Invalid Keychain profile name'
[[ "$IRIS_RELEASE_OUTPUT" == /* && "$IRIS_RELEASE_OUTPUT" != *$'\n'* ]] || fail 'Output must be an absolute path'
[[ ! -e "$IRIS_RELEASE_OUTPUT" && ! -L "$IRIS_RELEASE_OUTPUT" ]] || fail 'Output already exists; choose a new directory'
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project="$script_dir/Iris.xcodeproj"
[[ -d "$project" ]] || fail 'Iris.xcodeproj is missing'
if ! $execute; then
    printf 'Inputs valid. Dry run only: no identity/profile existence check, build, notarization, or files created.\n'
    exit 0
fi
[[ "$(uname -s)" == Darwin ]] || fail 'Packaging requires macOS and Xcode'
for tool in xcodebuild xcrun codesign security ditto spctl shasum plutil; do
    command -v "$tool" >/dev/null || fail "Missing tool: $tool"
done
identities=$(security find-identity -v -p codesigning)
printf '%s\n' "$identities" | grep -F -- "\"$IRIS_DEVELOPER_ID\"" >/dev/null || fail 'Valid Developer ID Application identity is unavailable'
xcrun --find notarytool >/dev/null
xcrun --find stapler >/dev/null
# mkdir is atomic: another run cannot claim the same output directory.
mkdir -- "$IRIS_RELEASE_OUTPUT"
output="$IRIS_RELEASE_OUTPUT"
archive="$output/Iris.xcarchive"
export_path="$output/export"
options="$output/ExportOptions.plist"
plutil -create xml1 "$options"
plutil -insert method -string developer-id "$options"
plutil -insert signingStyle -string manual "$options"
plutil -insert signingCertificate -string "$IRIS_DEVELOPER_ID" "$options"
plutil -insert teamID -string "$IRIS_TEAM_ID" "$options"
# No -allowProvisioningUpdates: credentials and certificates must already exist.
xcodebuild -project "$project" -scheme Iris -configuration Release \
    -destination 'generic/platform=macOS' -archivePath "$archive" \
    -derivedDataPath "$output/DerivedData" archive \
    CODE_SIGN_STYLE=Manual CODE_SIGN_IDENTITY="$IRIS_DEVELOPER_ID" \
    DEVELOPMENT_TEAM="$IRIS_TEAM_ID" ENABLE_HARDENED_RUNTIME=YES \
    ARCHS=arm64 ONLY_ACTIVE_ARCH=NO MACOSX_DEPLOYMENT_TARGET=15.0 \
    MARKETING_VERSION="$IRIS_RELEASE_VERSION" CURRENT_PROJECT_VERSION="$IRIS_RELEASE_BUILD"
xcodebuild -exportArchive -archivePath "$archive" \
    -exportPath "$export_path" -exportOptionsPlist "$options"
app="$export_path/Iris.app"
[[ -d "$app" ]] || fail 'Export did not produce Iris.app'
plist="$app/Contents/Info.plist"
read_plist() { /usr/libexec/PlistBuddy -c "Print :$1" "$plist"; }
[[ "$(read_plist CFBundleIdentifier)" == io.github.tky27.iris ]] || fail 'Unexpected bundle ID'
[[ "$(read_plist CFBundleShortVersionString)" == "$IRIS_RELEASE_VERSION" ]] || fail 'Marketing version mismatch'
[[ "$(read_plist CFBundleVersion)" == "$IRIS_RELEASE_BUILD" ]] || fail 'Build version mismatch'
[[ "$(read_plist LSMinimumSystemVersion)" == 15.0 ]] || fail 'Minimum macOS mismatch'
executable=$(read_plist CFBundleExecutable)
[[ "$executable" == Iris ]] || fail 'Unexpected bundle executable'
[[ "$(xcrun lipo -archs "$app/Contents/MacOS/$executable")" == arm64 ]] || fail 'Expected arm64-only executable'
[[ "$(xcrun vtool -show-build "$app/Contents/MacOS/$executable" | awk '/minos / {print $2}')" == 15.0 ]] || fail 'Mach-O minimum macOS mismatch'
codesign --verify --deep --strict --verbose=2 "$app"
signature=$(codesign -dv --verbose=4 "$app" 2>&1)
printf '%s\n' "$signature" | grep -Fx -- "Authority=$IRIS_DEVELOPER_ID" >/dev/null || fail 'Export is not signed with the requested Developer ID'
printf '%s\n' "$signature" | grep -Fx -- "TeamIdentifier=$IRIS_TEAM_ID" >/dev/null || fail 'Signature team mismatch'
printf '%s\n' "$signature" | grep -E 'flags=.*\(.*runtime.*\)' >/dev/null || fail 'Hardened runtime missing'
submission="$output/notary-submission.zip"
ditto -c -k --sequesterRsrc --keepParent "$app" "$submission"
xcrun notarytool submit "$submission" --keychain-profile "$IRIS_NOTARY_PROFILE" \
    --wait --output-format plist > "$output/notary-result.plist"
[[ "$(/usr/libexec/PlistBuddy -c 'Print :status' "$output/notary-result.plist")" == Accepted ]] || fail 'Notarization was not accepted; inspect notary-result.plist'
xcrun stapler staple "$app"
xcrun stapler validate "$app"
codesign --verify --deep --strict --verbose=2 "$app"
spctl --assess --type execute --verbose=2 "$app"
zip="$output/Iris-$IRIS_RELEASE_VERSION-$IRIS_RELEASE_BUILD-arm64.zip"
ditto -c -k --sequesterRsrc --keepParent "$app" "$zip"
(cd -- "$output" && shasum -a 256 "$(basename -- "$zip")" > "$(basename -- "$zip").sha256")
printf 'Local signed and notarized archive ready: %s\nNo release was uploaded.\n' "$zip"
