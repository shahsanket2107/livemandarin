#!/usr/bin/env bash
# Create a local self-signed code-signing certificate ("LiveMandarin Local") in the login
# keychain. build-app.sh signs the app with it, so macOS keeps the app's audio-capture
# permission across rebuilds (an ad-hoc signature changes every build and loses it).
# macOS may show a password prompt when the certificate is marked trusted.
set -euo pipefail
NAME="LiveMandarin Local"

if security find-identity -v -p codesigning | grep -q "$NAME"; then
  echo "Signing certificate '$NAME' already exists."
  exit 0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
openssl req -x509 -newkey rsa:2048 -keyout "$tmp/key.pem" -out "$tmp/cert.pem" -days 3650 -nodes \
  -subj "/CN=$NAME" -addext "keyUsage=digitalSignature" -addext "extendedKeyUsage=codeSigning" 2>/dev/null
openssl pkcs12 -export -legacy -inkey "$tmp/key.pem" -in "$tmp/cert.pem" -out "$tmp/id.p12" \
  -passout pass:tmp -name "$NAME" 2>/dev/null \
  || openssl pkcs12 -export -inkey "$tmp/key.pem" -in "$tmp/cert.pem" -out "$tmp/id.p12" -passout pass:tmp -name "$NAME"
security import "$tmp/id.p12" -k ~/Library/Keychains/login.keychain-db -P tmp -T /usr/bin/codesign -T /usr/bin/security
security add-trusted-cert -r trustRoot -p codeSign -k ~/Library/Keychains/login.keychain-db "$tmp/cert.pem"

security find-identity -v -p codesigning | grep -q "$NAME" && echo "Created signing certificate '$NAME'."
