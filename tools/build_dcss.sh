#!/usr/bin/env bash
# Build DCSS 0.17.1 outside the repository, with opt-in local pipe transport.
# The game and the derived patch are GPL-2.0-or-later. See THIRD_PARTY.md.
# Requirements: Linux, GNU make, GCC/G++, Perl, curl, tar, patch, sha256sum, tic.
# No root install, external game server, account, or network listener is used.
set -euo pipefail

ROOT="${1:-/tmp/fly-dcss017-runtime}"
JOBS="${JOBS:-4}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT"
ROOT="$(cd "$ROOT" && pwd)"
SRC="$ROOT/crawl-0.17.1/crawl-ref/source"
DEPS="$ROOT/deps"

fetch() {
    local filename="$1" url="$2" checksum="$3"
    if [[ ! -f "$ROOT/$filename" ]]; then
        curl --fail --location --retry 2 --output "$ROOT/$filename" "$url"
    fi
    echo "$checksum  $ROOT/$filename" | sha256sum --check --status || {
        echo "Checksum mismatch: $ROOT/$filename" >&2
        exit 1
    }
}

fetch crawl017.tar.gz \
    https://codeload.github.com/crawl/crawl/tar.gz/refs/tags/0.17.1 \
    e19645994bb90c15a1d114437afc906f730e5b0342a35270cb9a5a2b79143491
fetch lua.tar.gz \
    https://codeload.github.com/crawl/crawl-lua/tar.gz/881acfc2d9217f122147fbb5b21b2b141c4300c3 \
    9144b61e9b0c1b9c2d34846467f78a68ed2ea1b8269ff59eedee9e454e92fc67
fetch sqlite.tar.gz \
    https://codeload.github.com/crawl/crawl-sqlite/tar.gz/42f0cfa93c5bffc184ad7961c766c229aff8d16c \
    6ad8f08f1869dcf3617af8bcf8cffeaa314e0619d852c82b2bf5e043d0955d5f
fetch libpng.tar.gz \
    https://codeload.github.com/crawl/crawl-libpng/tar.gz/ccd6a270020a4d5acca57b285dbce69c2b787879 \
    98e52549e0d8fa0d597ba604cf527e69b2422f04f083218bae8f7a509fcbd034
fetch zlib.tar.gz \
    https://codeload.github.com/crawl/crawl-zlib/tar.gz/c34b4f48a59db2cd1c48eb2fea91b4a63dff28db \
    3c9b57eff29ccd818ce716db46095c18a15af24c551e3d16fc12f32969ac3918
fetch ncurses.tar.gz \
    https://ftp.gnu.org/gnu/ncurses/ncurses-6.5.tar.gz \
    136d91bc269a9a5785e5f9e980bc76ab57428f604ce3e5a5a90cebc767971cc6

if [[ ! -f "$SRC/Makefile" ]]; then
    tar -xzf "$ROOT/crawl017.tar.gz" -C "$ROOT"
fi
for dep in lua sqlite libpng zlib; do
    if [[ ! -f "$SRC/contrib/$dep/Makefile" && ! -f "$SRC/contrib/$dep/src/Makefile" ]]; then
        tar -xzf "$ROOT/$dep.tar.gz" --strip-components=1 -C "$SRC/contrib/$dep"
    fi
done
if [[ ! -f "$DEPS/lib/libncursesw.a" ]]; then
    [[ -d "$ROOT/ncurses-6.5" ]] || tar -xzf "$ROOT/ncurses.tar.gz" -C "$ROOT"
    (
        cd "$ROOT/ncurses-6.5"
        ./configure --prefix="$DEPS" --enable-widec --without-progs \
            --without-tests --without-cxx --without-cxx-binding --without-ada \
            --without-manpages
        make -j"$JOBS"
        make install.libs install.includes
    )
fi

# This private ncurses install searches its own terminfo directory by default.
# Compile the needed terminal entries from the same verified ncurses source.
mkdir -p "$DEPS/share/terminfo"
tic -x -e xterm,xterm-256color,vt100,linux -o "$DEPS/share/terminfo" \
    "$ROOT/ncurses-6.5/misc/terminfo.src"

# Standard release metadata normally supplied in Crawl's source distributions.
printf '0.17.1\n' > "$SRC/util/release_ver"
(
    cd "$ROOT/crawl-0.17.1"
    if patch --dry-run --forward -p1 < "$REPO/patches/dcss-0.17.1-pipes.patch" >/dev/null 2>&1; then
        patch --forward -p1 < "$REPO/patches/dcss-0.17.1-pipes.patch"
    elif ! patch --dry-run --reverse -p1 < "$REPO/patches/dcss-0.17.1-pipes.patch" >/dev/null 2>&1; then
        echo 'Pipe patch does not match this source tree' >&2
        exit 1
    fi
)
(
    cd "$SRC"
    # Explicit includes fix old transitive-header assumptions on modern GCC.
    # -O0 keeps this CPU-only smoke experiment's build short.
    CPATH="$DEPS/include${CPATH:+:$CPATH}" make -j"$JOBS" \
        WEBTILES=y BUILD_LUA=y BUILD_SQLITE=y BUILD_LIBPNG=y BUILD_ZLIB=y \
        NC_PREFIX="$DEPS" CFOPTIMIZE=-O0 \
        EXTERNAL_FLAGS='-D_GNU_SOURCE -include unistd.h' \
        EXTERNAL_FLAGS_L='-include bits/stdc++.h'
    ./crawl -version
)
printf '\nDCSS binary: %s/crawl\n' "$SRC"
