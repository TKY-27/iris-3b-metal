#!/bin/bash
set -euo pipefail

mode="${1:---verify}"
case "$mode" in run|--verify|--logs|--telemetry|--debug) ;; *) echo "Usage: $0 [run|--verify|--logs|--telemetry|--debug] [app arguments]" >&2; exit 2 ;; esac
if (( $# )); then shift; fi
root="$(cd "$(dirname "$0")/.." && pwd)"
development="${IRIS_DEV_ROOT:-$root/.scratch/macos-development}"
app="$development/IrisAudit.app"
mkdir -p "$development"

# Stop only this development bundle.
python3 - "$app/Contents/MacOS/Iris" <<'PY'
import os, signal, subprocess, sys, time
for line in subprocess.check_output(["ps", "-axo", "pid=,command="], text=True).splitlines():
    pid, command = line.strip().split(None, 1)
    if command == sys.argv[1] or command.startswith(sys.argv[1] + " "):
        os.kill(int(pid), signal.SIGTERM)
        for _ in range(100):
            try: os.kill(int(pid), 0)
            except ProcessLookupError: break
            time.sleep(.05)
        else: raise SystemExit("Development Iris is still running; close it before rebuilding.")
PY

xcodebuild -project "$root/apps/macos/Iris.xcodeproj" -scheme Iris -configuration Debug \
    -derivedDataPath "$development/DerivedData" build > "$development/build.log" 2>&1 || {
    tail -60 "$development/build.log"; exit 1;
}
python3 - "$development" <<'PY'
import pathlib, plistlib, shutil, sys
root = pathlib.Path(sys.argv[1]).resolve()
app = root / "IrisAudit.app"
shutil.copytree(root / "DerivedData/Build/Products/Debug/Iris.app", app, dirs_exist_ok=True)
path = app / "Contents/Info.plist"
info = plistlib.loads(path.read_bytes())
info.update(CFBundleIdentifier="io.github.tky27.iris.audit", CFBundleName="Iris Audit",
            IrisDevelopmentLibraryRoot=str(root / "Library"))
path.write_bytes(plistlib.dumps(info))
PY
codesign --force --deep --sign - "$app" >/dev/null 2>&1

if [[ "$mode" == --debug ]]; then
    exec lldb -- "$app/Contents/MacOS/Iris" "$@"
fi
/usr/bin/open -n "$app" --args "$@"
case "$mode" in
    --logs) exec /usr/bin/log stream --info --style compact --predicate 'process == "Iris"' ;;
    --telemetry) exec /usr/bin/log stream --info --style compact --predicate 'subsystem == "io.github.tky27.iris"' ;;
    --verify) sleep 1; pgrep -f "$app/Contents/MacOS/Iris" >/dev/null ;;
esac
echo "Opened $app with the isolated library at $development/Library"
