
"""
Experiment class patch to support optimization methods.

This script adapts the Experiment class to use the method registry
for instantiating optimization methods.
"""

import os
import sys
import yaml
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from methods import METHOD_REGISTRY, get_method_class

def fix_experiment_module():
    """
    Monkey patch the experiment module to use our method registry.
    
    This function modifies the Experiment class to use the method registry
    when creating optimization method instances.
    """
    experiment_file = os.path.join(project_root, "experiment.py")
    
    if not os.path.exists(experiment_file):
        print(f"Error: Could not find {experiment_file}")
        return False
    
    with open(experiment_file, 'r') as f:
        content = lines = f.readlines()
    
    backup_file = experiment_file + ".backup"
    if not os.path.exists(backup_file):
        with open(backup_file, 'w') as f:
            f.writelines(content)
        print(f"Created backup at {backup_file}")
    
    if import_section_end = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("import ") or line.strip().startswith("from "):
            import_section_end = i + 1
    
    if import_section_end > 0:
        lines.insert(import_section_end, "# Added by experiment_patch.py\n")
        lines.insert(import_section_end + 1, "from methods import METHOD_REGISTRY, get_method_class\n\n")
    
    method_creation_pattern = "Warning: No class specified for method"
    for i, line in enumerate(lines):
        if method_creation_pattern in line:
            # Add code to handle method classes using our registry
            # We'll look backwards to find the method_name variable
            method_name_line = -1
            for j in range(i-1, max(0, i-20), -1):
                if "method_name" in lines[j]:
                    method_name_line = j
                    break
            
            if method_name_line > 0:
                # Insert our method creation code
                patch_lines = [
                    "            # Added by experiment_patch.py\n",
                    "            method_class = get_method_class(method_name)\n",
                    "            if method_class:\n",
                    "                # Method class found in registry\n",
                    "                method = method_class(method_config)\n",
                    "                print(f\"Created {method_name} from registry\")\n",
                    "                continue  # Skip the warning\n",
                    "            # Original code follows if method not in registry\n"
                ]
                lines[i:i] = patch_lines
                break
    
    # Write the modified file
    with open(experiment_file, 'w') as f:
        f.writelines(lines)
    
    print(f"Patched {experiment_file} to use method registry")
    return True

if __name__ == "__main__":
    # Create the methods directories
    os.makedirs(os.path.join(project_root, "src", "methods"), exist_ok=True)
    
    # Fix the experiment module
    if fix_experiment_module():
        print("Patch applied successfully! You can now run your experiment.")
    else:
        print("Failed to apply patch.")