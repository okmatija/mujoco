#!/bin/bash
#
# Regenerates third_party/mujoco/copybara/{internal,public}_scrubs.patch.
#
# Fast rewrite of the original update_scrubs.sh:
#  - Runs only the scrub_internal/scrub_public copybara pipelines. The old
#    raw_internal/raw_public pipelines had no transformations, so their output
#    was a glob-filtered identity copy of the client; we instead copy pristine
#    content for exactly the scrubbed file set straight out of the client
#    (cache-warm after the scrub pipeline has read the same files).
#  - Diffs via a git index built from the scrubbed tree with --work-tree
#    pointed at the raw tree: no commit, no cp -r overlay, no full rescan.
#    Output format is identical to the old in-repo "git diff -U1".
#  - internal and public run as independent chains: the public diff no longer
#    waits for scrub_internal to finish.
#  - Scratch space lives on tmpfs (/dev/shm) when there is room.
#  - vcstool make-writable runs concurrently with the pipelines.

copybara="/google/bin/releases/copybara/public/copybara/copybara"
vcstool="/google/bin/releases/piper-fig/vcstool/vcstool"

# Copies, out of the client, pristine copies of exactly the files present in
# the scrubbed export. Equivalent to the old raw_* pipelines: those had no
# transformations, and files added/removed by scrubbing never appear in the
# patch (the old flow diffed tracked files only), so the scrubbed file list
# is the full set of paths a patch hunk can mention.
build_raw_dir() {
  local scrubbed="$1" raw="$2" client_root="$3"
  (cd "${scrubbed}" && find . \( -type f -o -type l \) -print0 \
      | sed -z 's|^\./||') > "${raw}.filelist" \
    && mkdir -p "${raw}" \
    && rsync -a --from0 --files-from="${raw}.filelist" "${client_root}/" "${raw}/"
}

# Emits the scrubbed -> raw diff in the same format as the old flow
# (index = scrubbed content, worktree = raw content). core.compression=0
# because the loose objects are throwaway.
emit_patch() {
  local scrubbed="$1" raw="$2" out="$3"
  local -
  set -o pipefail
  git -C "${scrubbed}" init -q \
    && git -C "${scrubbed}" -c core.compression=0 add . \
    && (cd "${raw}" && git --git-dir="${scrubbed}/.git" --work-tree=. diff -U1) \
       | { grep -v '^index ' || test $? -eq 1; } > "${out}"
}

# One end-to-end chain: copybara scrub pipeline, then raw copy, then diff.
run_chain() {
  local name="$1" citcroot="$2" scratch="$3"
  local scrubbed="${scratch}/${name}_scrubbed" raw="${scratch}/${name}_raw"
  mkdir -p "${scrubbed}" || return
  cd "${citcroot}" || return
  echo "[$(date +%T)] scrub_${name}: copybara pipeline"
  "${copybara}" third_party/mujoco/copy.bara.sky "scrub_${name}" ../ \
      --folder-dir="${scrubbed}" \
      --output-root="${scratch}/copybara_out_${name}" || return
  echo "[$(date +%T)] scrub_${name}: raw copy from client"
  build_raw_dir "${scrubbed}" "${raw}" "$(dirname "${citcroot}")" || return
  echo "[$(date +%T)] scrub_${name}: diff"
  emit_patch "${scrubbed}" "${raw}" "${scratch}/${name}_scrubs.patch" || return
  echo "[$(date +%T)] scrub_${name}: done"
}

# Stop here when sourced (tests exercise the functions above directly).
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then return 0; fi

source gbash.sh || exit

citcroot="$(gbash::get_google3_dir)"
retval="$?"
if [ "${retval}" != "0" ]; then
  echo "This script must be run from within a CitC client." 1>&2
  exit ${retval}
fi

if ! command -v rsync > /dev/null; then
  echo "ERROR: Install rsync and rerun this script." 1>&2
  exit 1
fi

# Prefer tmpfs for the scratch trees when it has headroom (>8G free).
if [ -w /dev/shm ] \
    && [ "$(df --output=avail -B1 /dev/shm | tail -n1 | tr -d ' ')" \
         -gt $((8 * 1024 * 1024 * 1024)) ]; then
  scratch="$(mktemp -d -p /dev/shm update_scrubs.XXXXXX)"
else
  scratch="$(mktemp -d)"
fi
retval="$?"
if [ "${retval}" != "0" ]; then
  exit ${retval}
fi
trap 'rm -rf "${scratch}"' EXIT

echo "Running scrub pipelines (logs: ${scratch}/{internal,public}.log)"
run_chain internal "${citcroot}" "${scratch}" \
    > "${scratch}/internal.log" 2>&1 &
pid_internal=$!
run_chain public "${citcroot}" "${scratch}" \
    > "${scratch}/public.log" 2>&1 &
pid_public=$!
(cd "${citcroot}" \
  && ${vcstool} make-writable "google3/third_party/mujoco/copybara/internal_scrubs.patch" \
  && ${vcstool} make-writable "google3/third_party/mujoco/copybara/public_scrubs.patch") \
    > "${scratch}/vcstool.log" 2>&1 &
pid_vcstool=$!

fail=0
for spec in "internal:${pid_internal}" "public:${pid_public}" \
            "vcstool:${pid_vcstool}"; do
  name="${spec%%:*}"
  if ! wait "${spec##*:}"; then
    echo "ERROR: ${name} step failed:" 1>&2
    cat "${scratch}/${name}.log" 1>&2
    fail=1
  fi
done
if [ "${fail}" != "0" ]; then
  exit 1
fi

cp "${scratch}/internal_scrubs.patch" \
   "${citcroot}/third_party/mujoco/copybara/internal_scrubs.patch"
cp "${scratch}/public_scrubs.patch" \
   "${citcroot}/third_party/mujoco/copybara/public_scrubs.patch"

echo "Updated scrub patches in $((SECONDS / 60))m$((SECONDS % 60))s."
