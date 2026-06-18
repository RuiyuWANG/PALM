#!/bin/bash

# Usage:
#   ./patch_manager.sh update
#   ./patch_manager.sh apply
#
# Description:
#   This script either updates/creates patches for Git directories
#   or applies patches to Git directories, depending on user input.

COMMAND="$1"

# Display usage if no argument is provided
if [ -z "$COMMAND" ]; then
  echo "Usage: $0 [update|apply]"
  exit 1
fi

confirm_action() {
  read -p "To confirm, please retype '$1': " CONFIRM
  if [[ "$CONFIRM" != "$1" ]]; then
    echo "Action $1 cancelled."
    exit 0
  fi
}

case "$COMMAND" in
  update)
    confirm_action "update"
    echo "Updating/creating patches..."
    # Create a hidden directory for patches if it doesn't exist
    mkdir -p .patches

    # Find all Git directories in the parent directory
    for dir in ../*/; do
      if [ -d "$dir/.git" ]; then
        submodule_name=$(basename "$dir")
        
        
        # Skip the palm directory
        if [ "$submodule_name" == "palm" ]; then
          continue
        fi
        
        # Generate the patch file name using the directory name
        patch_file="../palm/.patches/${submodule_name}.patch"
        
        # Navigate to the Git directory
        cd "$dir" || exit 1
        
        # Stage all changes (including new files)
        git add -A
        
        # Dump staged changes into a patch file
        git diff --cached --binary --full-index > "$patch_file"
        
        # Remove empty patch files
        if [ ! -s "$patch_file" ]; then
          rm -f "$patch_file"
        else
          echo "Patch created: $patch_file"
        fi
    
        # Undo the git add
        git reset
        
        # Return to the original directory
        cd - || exit 1
      fi
    done
    
    echo "Patches have been created in the .patches directory."
    ;;
    
  apply)
    confirm_action "apply"
    echo "Applying patches..."

    # This section applies each patch file in the `.patches` directory
    # to the matching Git directory based on the directory name.

    for patch_file in .patches/*.patch; do
      [ -e "$patch_file" ] || break

      # Extract the directory name from the patch file name
      dir_name=$(basename "$patch_file" .patch)

      # Check if a directory with the name exists
      if [ -d "../$dir_name" ]; then
        echo "Reverting directory: $dir_name to original state"
        cd "../$dir_name" || exit 1
        git checkout .
        echo "Applying patch to directory: $dir_name"
        git apply "../palm/.patches/$dir_name.patch"
        cd - || exit 1
      else
        echo "Directory $dir_name not found. Skipping..."
      fi
    done

    echo "All applicable patches have been applied."
    ;;
    
  *)
    echo "Invalid command: $COMMAND"
    echo "Usage: $0 [update|apply]"
    exit 1
    ;;
esac