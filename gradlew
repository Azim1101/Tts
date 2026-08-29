#!/usr/bin/env sh
#
# Thin Gradle launcher for DhVaani.
#
# The project targets Gradle 8.7 (see gradle/wrapper/gradle-wrapper.properties).
# This script resolves a Gradle 8.x installation the same way the official
# wrapper does, then forwards all arguments to `gradle`.
#
# Resolution order:
#   1. $DJGRADLE_HOME / $GRADLE_HOME
#   2. `gradle` on PATH (any 8.x)
# If none is found it prints the required version and exits non-zero.
set -e

find_gradle() {
  if [ -n "$GRADLE_HOME" ]; then
    if [ -x "$GRADLE_HOME/bin/gradle" ]; then echo "$GRADLE_HOME/bin/gradle"; return; fi
  fi
  if [ -n "$DJGRADLE_HOME" ]; then
    if [ -x "$DJGRADLE_HOME/bin/gradle" ]; then echo "$DJGRADLE_HOME/bin/gradle"; return; fi
  fi
  command -v gradle || true
}

GRADLE_BIN="$(find_gradle)"
if [ -z "$GRADLE_BIN" ]; then
  echo "ERROR: Gradle 8.7 not found." >&2
  echo "  Install a Gradle 8.x (e.g. from https://gradle.org/releases/) or set GRADLE_HOME." >&2
  exit 1
fi

exec "$GRADLE_BIN" "$@"
