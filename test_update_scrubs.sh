#!/bin/bash
# Byte-for-byte parity test: old update_scrubs.sh diff mechanics vs the new
# build_raw_dir + emit_patch functions, over a synthetic tree with edge cases.
set -u

work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT
cd "${work}" || exit 1

client="${work}/client"           # fake CitC client root (contains google3/)
tp="google3/third_party/mujoco"

# ---- build the fake client (raw content) ------------------------------------
mkdir -p "${client}/${tp}/src" "${client}/${tp}/.github/workflows" \
         "${client}/${tp}/dir with space" "${client}/${tp}/alpha"

# multi-hunk change candidate
{ for i in $(seq 1 40); do echo "line $i"; done; } > "${client}/${tp}/src/engine.cc"
sed -i -e '5a // copybara:strip_begin\n// internal secret\n// copybara:strip_end' \
       -e '30a // copybara:strip_begin\n// more internal\n// copybara:strip_end' \
       "${client}/${tp}/src/engine.cc"

echo "unchanged content" > "${client}/${tp}/src/unchanged.h"
printf 'name: CI\nsteps: internal-runner\n' > "${client}/${tp}/.github/workflows/ci.yml"
printf 'space file internal\n' > "${client}/${tp}/dir with space/b.txt"
printf 'whole file is internal\n' > "${client}/${tp}/src/deleted_by_scrub.cc"
printf '*.pyc\n' > "${client}/${tp}/.gitignore"
printf 'ignored raw bytes\n' > "${client}/${tp}/cache.pyc"
printf 'version=__COPYBARA_MUJOCO_VERSION__\n' > "${client}/${tp}/version.h"
printf '#!/bin/sh\necho internal tool\n' > "${client}/${tp}/tool.sh"
chmod +x "${client}/${tp}/tool.sh"
: > "${client}/${tp}/empty.txt"
# index sort-order stressors: '.'(0x2e) sorts before '/'(0x2f)
printf 'alpha txt internal\n' > "${client}/${tp}/alpha.txt"
printf 'alpha dir file internal\n' > "${client}/${tp}/alpha/file"

# ---- derive the scrubbed tree (what the scrub pipeline would output) --------
scrubbed_master="${work}/scrubbed_master"
mkdir -p "${scrubbed_master}"
cp -r "${client}/google3" "${scrubbed_master}/"
sm="${scrubbed_master}/${tp}"
sed -i '/copybara:strip_begin/,/copybara:strip_end/d' "${sm}/src/engine.cc"
sed -i 's/internal-runner/public-runner/' "${sm}/.github/workflows/ci.yml"
sed -i 's/internal/public/' "${sm}/dir with space/b.txt" "${sm}/tool.sh" \
       "${sm}/alpha.txt" "${sm}/alpha/file"
rm "${sm}/src/deleted_by_scrub.cc"                  # scrub deletes whole file
printf 'ignored scrubbed bytes\n' > "${sm}/cache.pyc"  # differs, but gitignored
sed -i 's/__COPYBARA_MUJOCO_VERSION__/3.2.1/' "${sm}/version.h"

# ---- OLD method: verbatim replica of the original script's git steps --------
old_scrubbed="${work}/old_scrubbed"
old_raw="${work}/old_raw"                # raw pipeline output = client content
mkdir -p "${old_scrubbed}" "${old_raw}"
cp -r "${scrubbed_master}/google3" "${old_scrubbed}/"
cp -r "${client}/google3" "${old_raw}/"

pushd "${old_scrubbed}" > /dev/null || exit 1
git init . -q
git add .
git -c user.email=t@t -c user.name=t commit -q -m "Original"
cp -r "${old_raw}"/* .
git diff -U1 | grep -v '^index ' > "${work}/old.patch"
popd > /dev/null || exit 1

# ---- NEW method: functions from the real script -----------------------------
source "$(dirname "$(readlink -f "$0")")/update_scrubs.sh"

new_scrubbed="${work}/new_scrubbed"
new_raw="${work}/new_raw"
mkdir -p "${new_scrubbed}"
cp -r "${scrubbed_master}/google3" "${new_scrubbed}/"

build_raw_dir "${new_scrubbed}" "${new_raw}" "${client}" || { echo "build_raw_dir FAILED"; exit 1; }
emit_patch "${new_scrubbed}" "${new_raw}" "${work}/new.patch" || { echo "emit_patch FAILED"; exit 1; }

# ---- compare ----------------------------------------------------------------
echo "--- old.patch: $(wc -l < "${work}/old.patch") lines, new.patch: $(wc -l < "${work}/new.patch") lines"
if cmp "${work}/old.patch" "${work}/new.patch"; then
  echo "PARITY OK: patches are byte-identical"
else
  echo "PARITY FAILED — diff follows:"
  diff "${work}/old.patch" "${work}/new.patch" | head -50
  exit 1
fi

# ---- sanity: the patch must reconstruct raw from scrubbed -------------------
apply_dir="${work}/apply"
mkdir -p "${apply_dir}"
cp -r "${scrubbed_master}/google3" "${apply_dir}/"
cd "${apply_dir}" || exit 1
if git apply -p1 "${work}/new.patch"; then
  echo "APPLY OK"
else
  echo "APPLY FAILED"
  exit 1
fi
# every file the patch touches must now equal client content
bad=0
while IFS= read -r rel; do
  if ! cmp -s "${apply_dir}/${rel}" "${client}/${rel}"; then
    echo "MISMATCH after apply: ${rel}"
    bad=1
  fi
done < <(sed -n 's|^+++ b/||p' "${work}/new.patch" | sed 's/\t$//')
[ "${bad}" = 0 ] && echo "ROUNDTRIP OK: applied tree matches client content"
exit "${bad}"
