"""
Directory Numeric Renamer
Renames all subdirectories in a given directory with numeric prefixes (001, 002, 003, etc.)
Preserves the current filesystem order of directories.

Usage:
    python directory_renamer.py [directory_path] [--replace]
    
Examples:
    python directory_renamer.py                    # Interactive mode (prefix mode)
    python directory_renamer.py /path/to/folder    # Direct path (prefix mode)
    python directory_renamer.py /path/to/folder --replace  # Replace names completely
    
Modes:
    Default (Prefix): 001 Original Name, 002 Another Folder, etc.
    Replace: 001, 002, 003, etc. (original names are lost)
"""

import os
import sys
from pathlib import Path

def get_subdirectories(directory_path):
    """
    Get all subdirectories in the given directory in filesystem order.
    
    Args:
        directory_path: Path to the parent directory
    
    Returns:
        list: List of subdirectory names in current order
    """
    subdirs = []
    
    try:
        # Get all items in directory
        items = os.listdir(directory_path)
        
        # Filter to only directories
        for item in items:
            item_path = os.path.join(directory_path, item)
            if os.path.isdir(item_path):
                subdirs.append(item)
    
    except Exception as e:
        print(f"Error reading directory: {e}")
        return []
    
    return subdirs

def calculate_padding(total_count):
    """
    Calculate the number of digits needed for zero-padding.
    
    Args:
        total_count: Total number of directories
    
    Returns:
        int: Number of digits for padding
    """
    if total_count == 0:
        return 1
    
    # Calculate digits needed
    digits = len(str(total_count))
    
    # Minimum 3 digits for aesthetic reasons
    return max(3, digits)

def preview_renaming(directory_path, subdirs, replace_mode=False):
    """
    Show a preview of what the renaming will look like.
    
    Args:
        directory_path: Path to the parent directory
        subdirs: List of subdirectory names
        replace_mode: If True, replace names completely; if False, add prefix
    
    Returns:
        list: List of (old_name, new_name) tuples
    """
    padding = calculate_padding(len(subdirs))
    renaming_plan = []
    
    mode_text = "REPLACE" if replace_mode else "PREFIX"
    print(f"\nRenaming Preview for: {directory_path}")
    print(f"Mode: {mode_text}")
    print("=" * 60)
    print(f"Total directories: {len(subdirs)}")
    print(f"Numeric padding: {padding} digits")
    print("=" * 60)
    
    for i, subdir in enumerate(subdirs, 1):
        numeric_prefix = str(i).zfill(padding)
        
        if replace_mode:
            # Replace mode: just use the numeric value
            new_name = numeric_prefix
        else:
            # Prefix mode: add numeric prefix to existing name
            # Check if directory already has numeric prefix
            if subdir.startswith(tuple(str(j).zfill(padding) for j in range(1, 1000))):
                # Remove existing numeric prefix
                # Find where the numeric prefix ends
                space_index = subdir.find(' ')
                underscore_index = subdir.find('_')
                dash_index = subdir.find('-')
                
                # Find the first separator after potential numeric prefix
                separators = [idx for idx in [space_index, underscore_index, dash_index] if idx > 0]
                
                if separators:
                    separator_index = min(separators)
                    base_name = subdir[separator_index + 1:]
                    new_name = f"{numeric_prefix} {base_name}"
                else:
                    # No separator found, might be just numbers
                    if subdir[:3].isdigit():
                        base_name = subdir[3:]
                        new_name = f"{numeric_prefix}{base_name}"
                    else:
                        new_name = f"{numeric_prefix} {subdir}"
            else:
                new_name = f"{numeric_prefix} {subdir}"
        
        renaming_plan.append((subdir, new_name))
        print(f"{i:>{padding}}. {subdir} → {new_name}")
    
    return renaming_plan

def rename_directories(directory_path, renaming_plan):
    """
    Execute the directory renaming plan.
    
    Args:
        directory_path: Path to the parent directory
        renaming_plan: List of (old_name, new_name) tuples
    
    Returns:
        tuple: (success_count, error_count)
    """
    success_count = 0
    error_count = 0
    
    print(f"\nExecuting renaming...")
    print("=" * 40)
    
    for old_name, new_name in renaming_plan:
        if old_name == new_name:
            print(f"  ↻ Skipped: {old_name} (no change needed)")
            continue
            
        old_path = os.path.join(directory_path, old_name)
        new_path = os.path.join(directory_path, new_name)
        
        try:
            # Check if target already exists
            if os.path.exists(new_path):
                print(f"  ✗ Error: {old_name} → Target '{new_name}' already exists")
                error_count += 1
                continue
            
            os.rename(old_path, new_path)
            print(f"  ✓ Renamed: {old_name} → {new_name}")
            success_count += 1
            
        except Exception as e:
            print(f"  ✗ Error renaming {old_name}: {e}")
            error_count += 1
    
    return success_count, error_count

def main():
    """Main function to run the directory renamer."""
    
    replace_mode = False
    directory_path = None
    
    # Parse command line arguments
    args = sys.argv[1:]
    
    # Check for --replace flag
    if '--replace' in args:
        replace_mode = True
        args.remove('--replace')
    
    # Get directory path
    if args:
        directory_path = args[0]
    else:
        directory_path = input("Enter the directory path: ").strip()
    
    # If not specified via command line, ask for mode interactively
    if '--replace' not in sys.argv:
        print("\nSelect renaming mode:")
        print("1. Prefix mode (001 Original Name, 002 Another Folder, etc.)")
        print("2. Replace mode (001, 002, 003, etc. - original names lost)")
        
        while True:
            mode_choice = input("Enter choice (1 or 2, default 1): ").strip()
            if not mode_choice or mode_choice == '1':
                replace_mode = False
                break
            elif mode_choice == '2':
                replace_mode = True
                break
            else:
                print("Please enter 1 or 2.")
    
    # Validate directory path
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        sys.exit(1)
    
    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a directory.")
        sys.exit(1)
    
    # Convert to absolute path
    directory_path = os.path.abspath(directory_path)
    
    mode_text = "REPLACE" if replace_mode else "PREFIX"
    print(f"\nDirectory Numeric Renamer")
    print(f"Target directory: {directory_path}")
    print(f"Mode: {mode_text}")
    
    # Get subdirectories
    subdirs = get_subdirectories(directory_path)
    
    if not subdirs:
        print("No subdirectories found.")
        return
    
    # Show preview
    renaming_plan = preview_renaming(directory_path, subdirs, replace_mode)
    
    # Ask for confirmation
    print(f"\n{'='*60}")
    print("CONFIRMATION REQUIRED")
    print(f"{'='*60}")
    
    if replace_mode:
        print("⚠️  WARNING: REPLACE mode will completely replace directory names!")
        print("   Original names will be lost and cannot be recovered!")
        print("   Directories will be renamed to: 001, 002, 003, etc.")
    else:
        print("Directories will be renamed with numeric prefixes added.")
        print("Original names will be preserved after the numbers.")
    
    print("\nThis operation cannot be easily undone.")
    
    while True:
        response = input("\nProceed with renaming? (Y/N): ").strip().upper()
        if response in ['Y', 'YES']:
            break
        elif response in ['N', 'NO']:
            print("Renaming cancelled.")
            return
        else:
            print("Please enter Y or N.")
    
    # Execute renaming
    success_count, error_count = rename_directories(directory_path, renaming_plan)
    
    # Show results
    print(f"\n{'='*40}")
    print("RENAMING COMPLETE")
    print(f"{'='*40}")
    print(f"Successfully renamed: {success_count}")
    print(f"Errors: {error_count}")
    
    if error_count > 0:
        print(f"\nSome directories could not be renamed. Please check the errors above.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)