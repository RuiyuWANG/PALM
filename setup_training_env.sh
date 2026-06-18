#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYREP_DIR="${PYREP_DIR:-$WORKSPACE_DIR/pyrep}"
RLBENCH_DIR="${RLBENCH_DIR:-$WORKSPACE_DIR/rlbench}"
PALM_DIR="$SCRIPT_DIR"

ENV_NAME="${ENV_NAME:-palm}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
RESET_ENV="${RESET_ENV:-0}"
CLONE_DEPS="${CLONE_DEPS:-1}"
PYREP_URL="${PYREP_URL:-https://github.com/stepjam/PyRep.git}"
PYREP_REF="${PYREP_REF:-8f420be8064b1970aae18a9cfbc978dfb15747ef}"
RLBENCH_URL="${RLBENCH_URL:-https://github.com/stepjam/RLBench.git}"
RLBENCH_REF="${RLBENCH_REF:-7c3f425f4a0b6b5ce001ba7246354eb3c70555be}"
DATA_DIR="${DATA_DIR:-$WORKSPACE_DIR/data}"
BACKGROUND_DIR="${BACKGROUND_DIR:-$DATA_DIR/backgrounds}"
BACKGROUND_URL="${BACKGROUND_URL:-http://images.cocodataset.org/zips/val2017.zip}"
BACKGROUND_LIMIT="${BACKGROUND_LIMIT:-512}"
APPLY_PATCHES="${APPLY_PATCHES:-1}"
RESET_PATCH_TARGETS="${RESET_PATCH_TARGETS:-0}"
SKIP_BACKGROUNDS="${SKIP_BACKGROUNDS:-0}"
INSTALL_PYTORCH="${INSTALL_PYTORCH:-1}"
TORCH_PACKAGES="${TORCH_PACKAGES:-torch==2.5.1 torchvision==0.20.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
INSTALL_PYREP_REQUIREMENTS="${INSTALL_PYREP_REQUIREMENTS:-1}"
INSTALL_RLBENCH_REQUIREMENTS="${INSTALL_RLBENCH_REQUIREMENTS:-1}"

log() {
  printf "\n[setup-training-env] %s\n" "$*"
}

die() {
  printf "\n[setup-training-env] ERROR: %s\n" "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

detect_coppeliasim_root() {
  if [ -n "${COPPELIASIM_ROOT:-}" ] && [ -f "$COPPELIASIM_ROOT/libcoppeliaSim.so" ]; then
    printf "%s\n" "$COPPELIASIM_ROOT"
    return
  fi

  local candidate
  for candidate in \
    "$HOME"/Simulators/CoppeliaSim* \
    "$HOME"/CoppeliaSim* \
    /opt/CoppeliaSim* \
    /usr/local/CoppeliaSim*; do
    if [ -f "$candidate/libcoppeliaSim.so" ]; then
      printf "%s\n" "$candidate"
      return
    fi
  done

  local found
  found="$(find "$HOME" -maxdepth 5 -type f -name libcoppeliaSim.so 2>/dev/null | head -n 1 || true)"
  if [ -n "$found" ]; then
    dirname "$found"
    return
  fi

  die "Could not find CoppeliaSim. Set COPPELIASIM_ROOT=/path/to/CoppeliaSim_Edu_V4_1_0_* and rerun."
}

setup_pyrep() {
  export COPPELIASIM_ROOT
  case ":${LD_LIBRARY_PATH:-}:" in
    *:"$COPPELIASIM_ROOT":*) ;;
    *) export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$COPPELIASIM_ROOT" ;;
  esac
  export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
}

ensure_dependency_checkout() {
  local name="$1"
  local dir="$2"
  local url="$3"
  local ref="$4"

  if [ -d "$dir/.git" ]; then
    log "Using existing $name checkout: $dir"
    return
  fi

  if [ -e "$dir" ]; then
    die "$name path exists but is not a git checkout: $dir"
  fi

  [ "$CLONE_DEPS" = "1" ] || die "Missing $name checkout: $dir. Set CLONE_DEPS=1 or clone $url manually."

  require_command git
  log "Cloning $name from $url into $dir"
  git clone "$url" "$dir"
  log "Checking out $name ref $ref"
  git -C "$dir" checkout "$ref"
}

ensure_dependency_checkouts() {
  ensure_dependency_checkout "PyRep" "$PYREP_DIR" "$PYREP_URL" "$PYREP_REF"
  ensure_dependency_checkout "RLBench" "$RLBENCH_DIR" "$RLBENCH_URL" "$RLBENCH_REF"
}

apply_dependency_patches() {
  [ "$APPLY_PATCHES" = "1" ] || {
    log "Skipping dependency patches because APPLY_PATCHES=$APPLY_PATCHES"
    return
  }

  log "Applying dependency patches from $PALM_DIR/.patches"
  shopt -s nullglob
  local patch_file target_name target_dir
  for patch_file in "$PALM_DIR"/.patches/*.patch; do
    target_name="$(basename "$patch_file" .patch)"
    target_dir="$WORKSPACE_DIR/$target_name"
    [ -d "$target_dir/.git" ] || {
      log "Skipping $target_name: $target_dir is not a git checkout"
      continue
    }

    if git -C "$target_dir" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
      log "$target_name patch already applied"
      continue
    fi

    if [ "$RESET_PATCH_TARGETS" = "1" ]; then
      log "Resetting tracked files in $target_name before patching"
      git -C "$target_dir" checkout .
    fi

    git -C "$target_dir" apply --check "$patch_file" || die "Patch does not apply cleanly: $patch_file"
    git -C "$target_dir" apply "$patch_file"
    log "Applied $target_name patch"
  done
  shopt -u nullglob
}

create_and_activate_conda_env() {
  require_command conda

  local conda_base
  conda_base="$(conda info --base)"
  # shellcheck disable=SC1091
  source "$conda_base/etc/profile.d/conda.sh"

  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    if [ "$RESET_ENV" = "1" ]; then
      log "Removing existing conda env because RESET_ENV=1: $ENV_NAME"
      conda env remove -y -n "$ENV_NAME"
      log "Creating conda env: $ENV_NAME (python=$PYTHON_VERSION)"
      conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
    else
      log "Using existing conda env: $ENV_NAME"
      log "Set RESET_ENV=1 to recreate it from scratch if stale editable packages are present"
    fi
  else
    log "Creating conda env: $ENV_NAME (python=$PYTHON_VERSION)"
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
  fi

  conda activate "$ENV_NAME"
  log "Active Python: $(python -c 'import sys; print(sys.executable)')"
}

write_activation_hooks() {
  local activate_dir="$CONDA_PREFIX/etc/conda/activate.d"
  local deactivate_dir="$CONDA_PREFIX/etc/conda/deactivate.d"
  mkdir -p "$activate_dir" "$deactivate_dir"

  cat > "$activate_dir/palm_training_env.sh" <<EOF
export PALM_WORKSPACE_DIR="$WORKSPACE_DIR"
export _PALM_OLD_LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:-}"
setup_pyrep() {
  export COPPELIASIM_ROOT="$COPPELIASIM_ROOT"
  case ":\${LD_LIBRARY_PATH:-}:" in
    *:"\$COPPELIASIM_ROOT":*) ;;
    *) export LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:+\$LD_LIBRARY_PATH:}\$COPPELIASIM_ROOT" ;;
  esac
  export QT_QPA_PLATFORM_PLUGIN_PATH="\$COPPELIASIM_ROOT"
}
setup_pyrep
export PYTHONNOUSERSITE=1
EOF

  cat > "$deactivate_dir/palm_training_env.sh" <<'EOF'
if [ -n "${_PALM_OLD_LD_LIBRARY_PATH+x}" ]; then
  export LD_LIBRARY_PATH="$_PALM_OLD_LD_LIBRARY_PATH"
  unset _PALM_OLD_LD_LIBRARY_PATH
fi
unset PALM_WORKSPACE_DIR
unset COPPELIASIM_ROOT
unset QT_QPA_PLATFORM_PLUGIN_PATH
unset PYTHONNOUSERSITE
unset -f setup_pyrep 2>/dev/null || true
EOF

  export PALM_WORKSPACE_DIR="$WORKSPACE_DIR"
  setup_pyrep
  export PYTHONNOUSERSITE=1
  log "Wrote conda activation hooks for CoppeliaSim and PALM"
}

remove_stale_editable_package() {
  python - "$1" "$2" <<'PY'
import json
import os
import shutil
import sys
from pathlib import Path
from importlib import metadata

package_name = sys.argv[1]
expected_path = Path(sys.argv[2]).resolve()

try:
    dist = metadata.distribution(package_name)
except metadata.PackageNotFoundError:
    raise SystemExit(0)

direct_url_path = Path(dist.locate_file("direct_url.json"))
if not direct_url_path.exists():
    raise SystemExit(0)

try:
    direct_url = json.loads(direct_url_path.read_text())
except json.JSONDecodeError:
    raise SystemExit(0)

url = direct_url.get("url", "")
if not url.startswith("file://"):
    raise SystemExit(0)

installed_path = Path(url[7:]).resolve()
if installed_path == expected_path:
    raise SystemExit(0)

print(
    f"Removing stale editable metadata for {package_name}: "
    f"{installed_path} -> {expected_path}"
)

dist_info = Path(getattr(dist, "_path"))
files = list(dist.files or [])
for rel_path in files:
    path = Path(dist.locate_file(rel_path))
    name = path.name.lower()
    if name.startswith("__editable__") and path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    elif name.endswith(".egg-link") and path.exists():
        path.unlink()

if dist_info.exists():
    shutil.rmtree(dist_info)
PY
}

install_common_python_packages() {
  log "Installing Python packaging tools"
  python -m pip install --upgrade pip setuptools wheel

  if [ "$INSTALL_PYTORCH" = "1" ]; then
    if [ -n "$TORCH_INDEX_URL" ]; then
      log "Installing PyTorch packages from $TORCH_INDEX_URL: $TORCH_PACKAGES"
      python -m pip install --force-reinstall $TORCH_PACKAGES --index-url "$TORCH_INDEX_URL"
    else
      log "Installing PyTorch packages from the default pip index: $TORCH_PACKAGES"
      python -m pip install --force-reinstall $TORCH_PACKAGES
    fi
  fi
}

install_pyrep() {
  setup_pyrep
  log "Installing PyRep requirements and editable package"
  if [ "$INSTALL_PYREP_REQUIREMENTS" = "1" ]; then
    python -m pip install -r "$PYREP_DIR/requirements.txt"
  fi
  remove_stale_editable_package PyRep "$PYREP_DIR"
  python -m pip install -e "$PYREP_DIR"
  log "Building PyRep CFFI extension in place"
  (cd "$PYREP_DIR" && python setup.py build_ext --inplace)
}

install_rlbench() {
  setup_pyrep
  log "Installing RLBench requirements and editable package"
  if [ "$INSTALL_RLBENCH_REQUIREMENTS" = "1" ]; then
    python -m pip install -r "$RLBENCH_DIR/requirements.txt"
  fi
  remove_stale_editable_package rlbench "$RLBENCH_DIR"
  python -m pip install -e "$RLBENCH_DIR"
}

install_palm() {
  log "Installing PALM in editable mode"
  remove_stale_editable_package palm "$PALM_DIR"
  python -m pip install -e "$PALM_DIR"
}

download_backgrounds() {
  [ "$SKIP_BACKGROUNDS" = "1" ] && {
    log "Skipping background download because SKIP_BACKGROUNDS=1"
    return
  }

  mkdir -p "$DATA_DIR" "$BACKGROUND_DIR"

  local existing_count
  existing_count="$(find "$BACKGROUND_DIR" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)"
  if [ "$existing_count" -gt 0 ]; then
    log "Background directory already has $existing_count images: $BACKGROUND_DIR"
    return
  fi

  log "Downloading background images to $BACKGROUND_DIR"
  python - "$BACKGROUND_URL" "$DATA_DIR" "$BACKGROUND_DIR" "$BACKGROUND_LIMIT" <<'PY'
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

url = sys.argv[1]
data_dir = Path(sys.argv[2])
background_dir = Path(sys.argv[3])
limit = int(sys.argv[4])
suffixes = {".jpg", ".jpeg", ".png"}

data_dir.mkdir(parents=True, exist_ok=True)
background_dir.mkdir(parents=True, exist_ok=True)

archive = data_dir / Path(url.split("?")[0]).name
if not archive.exists():
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, archive)
else:
    print(f"Using existing archive {archive}")

count = 0
if zipfile.is_zipfile(archive):
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            src_name = Path(member.filename)
            if member.is_dir() or src_name.suffix.lower() not in suffixes:
                continue
            dst = background_dir / src_name.name
            if not dst.exists():
                with zf.open(member) as src, dst.open("wb") as out:
                    shutil.copyfileobj(src, out)
            count += 1
            if limit > 0 and count >= limit:
                break
else:
    if archive.suffix.lower() not in suffixes:
        raise RuntimeError(f"Unsupported background archive/file: {archive}")
    shutil.copy2(archive, background_dir / archive.name)
    count = 1

print(f"Prepared {count} background images in {background_dir}")
PY
}

smoke_test_imports() {
  log "Running import smoke tests"
  CONFIG_PATH="$PALM_DIR/palm/configs/train/lift_spam_config.json" python - <<'PY'
import os

import palm
import pyrep
import rlbench
from palm.utils.net_utils import parse_network_configs

cfg = parse_network_configs(os.environ["CONFIG_PATH"])
print(f"PALM import OK; lift_spam shapes: low_dim={cfg.network.low_dim_shape}, action={cfg.network.action_shape}")
PY
}

main() {
  [ -d "$PALM_DIR/palm" ] || die "Missing PALM package directory: $PALM_DIR/palm"

  ensure_dependency_checkouts

  COPPELIASIM_ROOT="$(detect_coppeliasim_root)"
  export COPPELIASIM_ROOT
  setup_pyrep
  log "Using COPPELIASIM_ROOT=$COPPELIASIM_ROOT"

  apply_dependency_patches
  create_and_activate_conda_env
  write_activation_hooks
  install_common_python_packages
  install_pyrep
  install_rlbench
  install_palm
  download_backgrounds
  smoke_test_imports

  cat <<EOF

Training environment is ready.

Next shell:
  conda activate $ENV_NAME
  cd "$PALM_DIR"
  palm-train -c palm/configs/train/lift_spam_config.json

Useful options:
  ENV_NAME=$ENV_NAME
  DATA_DIR=$DATA_DIR
  BACKGROUND_DIR=$BACKGROUND_DIR
  BACKGROUND_LIMIT=$BACKGROUND_LIMIT
  RESET_ENV=$RESET_ENV
  CLONE_DEPS=$CLONE_DEPS
EOF
}

main "$@"
