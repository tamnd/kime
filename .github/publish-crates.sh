#!/usr/bin/env bash
# Publishes every crate in the workspace to crates.io at the workspace version, and can be run again to finish what an earlier run started.
#
# `cargo publish --workspace` already works out the order and waits for the index between crates. What it does not do is cope with the crates.io rate limits. A crate that has never been published gets a burst of 5 and then one every ten minutes, and an existing crate gets a burst of 30 and then one a minute. The first release of kime is 15 new crates, so it hits the first limit after 5 and takes about an hour and forty minutes, mostly waiting. When cargo gets a 429 it stops, and a second call stops again at the first crate that is already up. So this script asks the index what is there, excludes it, and sleeps for as long as the limit in play asks for.
#
# The token comes from CARGO_REGISTRY_TOKEN. When that is empty and KIME_ENV_FILE points at a file of `export NAME=value` lines, the one variable is read out of it in a subshell, so nothing else from that file ends up in this process and the token never appears on a command line.
#
# Run it from the root of the workspace. When everything is already up it prints so and exits zero.

set -euo pipefail

if [ -z "${CARGO_REGISTRY_TOKEN:-}" ] && [ -n "${KIME_ENV_FILE:-}" ]; then
  # shellcheck disable=SC1090
  CARGO_REGISTRY_TOKEN=$(. "$KIME_ENV_FILE" >/dev/null 2>&1; printf '%s' "${CARGO_REGISTRY_TOKEN:-}")
  export CARGO_REGISTRY_TOKEN
fi
if [ -z "${CARGO_REGISTRY_TOKEN:-}" ]; then
  echo "CARGO_REGISTRY_TOKEN is empty, so nothing can be published" >&2
  exit 1
fi

# Ten minutes and ten seconds for a new crate, seventy seconds for an existing one. The extra ten seconds cover clock skew between here and the registry.
new_pause=610
existing_pause=70

metadata=$(cargo metadata --format-version 1 --no-deps)
version=$(echo "$metadata" | jq -r '.packages[] | select(.name == "kime") | .version')
# xtask says `publish = false`, which shows up here as an empty list of registries.
crates=$(echo "$metadata" | jq -r '.packages[] | select(.publish != []) | .name' | sort)
total=$(echo "$crates" | wc -w | tr -d ' ')

# The sparse index path for a crate name, which every registry client computes the same way.
index_path() {
  local name=$1
  case ${#name} in
    1) echo "1/$name" ;;
    2) echo "2/$name" ;;
    3) echo "3/${name:0:1}/$name" ;;
    *) echo "${name:0:2}/${name:2:2}/$name" ;;
  esac
}

index_entry() {
  curl --silent --fail "https://index.crates.io/$(index_path "$1")" 2>/dev/null || true
}

log=$(mktemp)
trap 'rm -f "$log"' EXIT
attempts=$((total + 5))

for attempt in $(seq 1 "$attempts"); do
  exclude=()
  up=0
  brand_new=0
  for crate in $crates; do
    entry=$(index_entry "$crate")
    if echo "$entry" | grep -q "\"vers\":\"$version\""; then
      exclude+=(--exclude "$crate")
      up=$((up + 1))
    elif [ -z "$entry" ]; then
      brand_new=$((brand_new + 1))
    fi
  done

  if [ "$up" -eq "$total" ]; then
    echo "all $total crates are on crates.io at $version"
    exit 0
  fi

  if [ "$brand_new" -gt 0 ]; then
    pause=$new_pause
  else
    pause=$existing_pause
  fi
  echo "attempt $attempt: $up of $total up at $version, $brand_new never published before"

  # The odd expansion is for bash 3.2 on macOS, which treats an empty array as unbound under `set -u`.
  if cargo publish --workspace --locked ${exclude[@]+"${exclude[@]}"} >"$log" 2>&1; then
    cat "$log"
    echo "published the remaining $((total - up)) crates at $version"
    exit 0
  fi
  tail -5 "$log"

  # Another run, or the release job, got there first. The index is read again at the top of the loop.
  if grep -q "already exists on crates.io index" "$log"; then
    sleep 10
    continue
  fi
  if grep -q "too many versions of this crate in the last 24 hours" "$log"; then
    echo "a crate is at the limit of twenty versions a day, run this again tomorrow" >&2
    exit 1
  fi
  if ! grep -q "429 Too Many Requests" "$log"; then
    echo "the publish failed for a reason that waiting will not fix" >&2
    exit 1
  fi
  echo "rate limited, waiting ${pause}s"
  sleep "$pause"
done

echo "gave up after $attempts attempts, run this again and it carries on from the index" >&2
exit 1
