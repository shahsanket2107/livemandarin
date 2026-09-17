#!/usr/bin/env bash
# Build the native menu-bar app into build/LiveMandarin.app (ad-hoc signed for local use).
# Uses swiftc directly (no SwiftPM) and picks the newest SDK the installed compiler accepts.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$ROOT/build/LiveMandarin.app"
mkdir -p "$ROOT/build"

pick_sdk() {
  local probe; probe="$(mktemp -d)"; echo 'let x = 1' > "$probe/p.swift"
  for sdk in $(ls -d /Library/Developer/CommandLineTools/SDKs/MacOSX*.*.sdk 2>/dev/null | sort -rV); do
    if swiftc -sdk "$sdk" -typecheck "$probe/p.swift" >/dev/null 2>&1; then echo "$sdk"; rm -rf "$probe"; return; fi
  done
  rm -rf "$probe"; echo "No SDK matches the installed Swift compiler (try: xcode-select --install)" >&2; exit 1
}
SDK="$(pick_sdk)"
echo "SDK: $SDK"

swiftc -O -parse-as-library -sdk "$SDK" -target arm64-apple-macos14.0 \
  -framework AppKit -framework SwiftUI -framework ScreenCaptureKit -framework ServiceManagement -framework AVFoundation \
  "$ROOT"/app/Sources/LiveMandarin/*.swift -o "$ROOT/build/LiveMandarin"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
mv "$ROOT/build/LiveMandarin" "$APP/Contents/MacOS/LiveMandarin"
sed "s|__SERVER_DIR__|$ROOT|" "$ROOT/app/Info.plist" > "$APP/Contents/Info.plist"
[ -f "$ROOT/app/AppIcon.icns" ] && cp "$ROOT/app/AppIcon.icns" "$APP/Contents/Resources/"
# Sign with the local certificate if present (keeps macOS permissions across rebuilds), else ad-hoc.
IDENTITY="LiveMandarin Local"
if security find-identity -v -p codesigning | grep -q "$IDENTITY"; then
  codesign --force --sign "$IDENTITY" "$APP" >/dev/null
else
  echo "note: no '$IDENTITY' certificate; ad-hoc signing (permissions reset on every rebuild)"
  codesign --force --sign - "$APP" >/dev/null
fi

# Install to /Applications so Spotlight/Launchpad can find it; the running copy is replaced.
pkill -x LiveMandarin 2>/dev/null || true
rm -rf "/Applications/LiveMandarin.app"
ditto "$APP" "/Applications/LiveMandarin.app"
echo "Installed: /Applications/LiveMandarin.app"
