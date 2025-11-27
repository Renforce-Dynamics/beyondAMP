#!/bin/bash
set -e

# ============================================================
# Utility functions
# ============================================================

# Clone a repository if it doesn't already exist
# Tries SSH first, falls back to HTTPS if SSH fails
clone_repo() {
    local repo_url=$1
    local target_dir=$2

    if [ ! -d "$target_dir/.git" ]; then
        echo "→ Cloning $repo_url to $target_dir ..."
        mkdir -p "$(dirname "$target_dir")"
        
        # Try SSH first
        if git clone "$repo_url" "$target_dir" 2>/dev/null; then
            echo "✔ Successfully cloned via SSH"
        else
            # Convert SSH URL to HTTPS if SSH failed
            local https_url=$(echo "$repo_url" | sed 's|git@github.com:|https://github.com/|' | sed 's|\.git$||')
            echo "⚠ SSH clone failed, trying HTTPS: $https_url"
            if git clone "$https_url.git" "$target_dir" 2>/dev/null; then
                echo "✔ Successfully cloned via HTTPS"
            else
                echo "❌ Failed to clone repository. Please check:"
                echo "   1. Repository exists and is accessible"
                echo "   2. You have proper access rights"
                echo "   3. Network connectivity"
                exit 1
            fi
        fi
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
    "./source/beyondMimic"
    "./source/beyondAMP"
)

install_modules "${modules[@]}"

echo "✅ All done!"
