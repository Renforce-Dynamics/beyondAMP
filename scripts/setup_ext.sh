#!/bin/bash
set -e

# ============================================================
# Utility functions
# ============================================================

# Clone a repository if it doesn't already exist
clone_repo() {
    local repo_url=$1
    local target_dir=$2

    if [ ! -d "$target_dir/.git" ]; then
        echo "→ Cloning $repo_url to $target_dir ..."
        mkdir -p "$(dirname "$target_dir")"
        git clone "$repo_url" "$target_dir"
    else
        echo "✔ Repository already exists at $target_dir, skipping clone."
    fi
}

# Check and install local editable Python packages
install_modules() {
    local modules=("$@")

    echo "📦 Installing local editable modules..."
    for module in "${modules[@]}"; do
        if [ -d "$module" ]; then
            echo "→ Installing $module ..."
            pip install -e "$module"
        else
            echo "⚠ Skipped $module (directory not found)"
        fi
    done
}

# ============================================================
# Main script
# ============================================================

echo "🧩 Checking and cloning required repositories..."

clone_repo git@github.com:Renforce-Dynamics/assetslib.git ./data/assets/assetslib
clone_repo git@github.com:Renforce-Dynamics/robotlib.git ./source/robotlib

modules=(
    "./source/robotlib"
    "./source/rsl_rl_amp"
    "./source/amp_tasks"
    "./source/beyondAMP"
    "./source/third_party/beyondMimic"
    "./source/third_party/sim2simlib"
)

install_modules "${modules[@]}"

echo "✅ All done!"
