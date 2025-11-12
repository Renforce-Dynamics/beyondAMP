import os
import json

# =========================================================
# VSCode Configuration Utility
# ---------------------------------------------------------
# This script updates or creates a .vscode/settings.json file
# and ensures that all relative source paths are properly added
# to "python.analysis.extraPaths" for IntelliSense and debugging.
# =========================================================

# Determine the absolute path of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.join(script_dir, "..")

# Path to the VSCode settings file
settings_path = os.path.join(repo_dir, ".vscode", "settings.json")

# Relative source paths to include (relative to this script)
relative_extra_paths = [
    "./source/fighterLab",
    "./source/fighterTask",
    "./source/fsmLab",
    "./source/robotlib",
    "./source/rsl_rl",
    "./source/third_party/beyondMimic",
    "./source/third_party/locomotion_rl_lab",
]

# Normalize and convert relative paths to absolute paths
absolute_extra_paths = [
    os.path.normpath(os.path.join(repo_dir, rel_path))
    for rel_path in relative_extra_paths
]

# =========================================================
# Load or initialize settings
# =========================================================

if not os.path.exists(settings_path):
    os.makedirs(os.path.dirname(settings_path), exist_ok=True)
    settings = {}
    print(f"[INFO] Created new settings.json at {settings_path}")
else:
    with open(settings_path, "r") as f:
        settings = json.load(f)

# Ensure the structure for python.analysis.extraPaths exists
if "python.analysis.extraPaths" not in settings:
    settings["python.analysis.extraPaths"] = {}

extra_paths = settings["python.analysis.extraPaths"]

# =========================================================
# Update extraPaths without duplicating entries
# =========================================================

for path in absolute_extra_paths:
    if path not in extra_paths:
        extra_paths.append(path)
        print(f"[ADDED] {path}")
    else:
        print(f"[SKIPPED] Already exists: {path}")

# =========================================================
# Save updated configuration
# =========================================================

with open(settings_path, "w") as f:
    json.dump(settings, f, indent=4)

print(f"[DONE] VSCode settings successfully updated: {settings_path}")
