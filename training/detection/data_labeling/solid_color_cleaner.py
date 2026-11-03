"""
Solid Color Image Detector and Cleaner
Scans folders for images that are essentially one solid color and marks them for deletion.
User can specify tolerance percentage for color variation detection.

Usage:
    python solid_color_cleaner.py [folder_path] [tolerance_percent]
    
Examples:
    python solid_color_cleaner.py                    # Interactive mode
    python solid_color_cleaner.py /path/to/folder    # Interactive tolerance
    python solid_color_cleaner.py /path/to/folder 2.5   # 2.5% tolerance
"""

import os
import sys
from PIL import Image, ImageStat
import numpy as np
from pathlib import Path

# Supported image extensions
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}

def is_solid_color(image_path, tolerance_percent=1.0):
    """
    Check if an image consists of essentially one solid color.
    
    Args:
        image_path: Path to the image file
        tolerance_percent: Tolerance percentage for color variation (default 1%)
    
    Returns:
        bool: True if image is solid color, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Get image statistics
            stat = ImageStat.Stat(img)
            
            # Calculate the standard deviation for each channel (R, G, B)
            # Low standard deviation indicates uniform color
            std_devs = stat.stddev
            
            # Tolerance: 1% of 255 = 2.55
            tolerance = 255 * (tolerance_percent / 100)
            
            # Check if all channels have low standard deviation
            is_uniform = all(std_dev <= tolerance for std_dev in std_devs)
            
            if is_uniform:
                print(f"  → Solid color detected: {image_path}")
                return True
            
            return False
            
    except Exception as e:
        print(f"  → Error processing {image_path}: {e}")
        return False

def find_and_mark_solid_images(root_folder, tolerance_percent=1.0):
    """
    Find all solid color images in the folder tree and mark them for deletion.
    
    Args:
        root_folder: Root directory to scan
        tolerance_percent: Tolerance percentage for color variation
    
    Returns:
        list: List of marked files
    """
    marked_files = []
    
    print(f"Scanning folder: {root_folder}")
    print("=" * 50)
    
    for root, dirs, files in os.walk(root_folder):
        print(f"\nChecking directory: {root}")
        
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            # Check if it's an image file
            if file_ext in SUPPORTED_EXTENSIONS:
                print(f"  Checking: {file}")
                
                if is_solid_color(file_path, tolerance_percent):
                    # Create new filename with deletion flag
                    new_filename = f"[for deletion]{file}"
                    new_path = os.path.join(root, new_filename)
                    
                    try:
                        os.rename(file_path, new_path)
                        marked_files.append(new_path)
                        print(f"  ✓ Marked: {new_filename}")
                    except Exception as e:
                        print(f"  ✗ Error renaming {file}: {e}")
    
    return marked_files

def delete_marked_files(root_folder):
    """
    Delete all files marked with [for deletion] prefix.
    
    Args:
        root_folder: Root directory to scan
    """
    deleted_count = 0
    
    print(f"\nDeleting marked files in: {root_folder}")
    print("=" * 50)
    
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.startswith("[for deletion]"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    print(f"  ✓ Deleted: {file}")
                except Exception as e:
                    print(f"  ✗ Error deleting {file}: {e}")
    
    print(f"\nTotal files deleted: {deleted_count}")

def main():
    """Main function to run the solid color image cleaner."""
    
    # Get folder path from user
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = input("Enter the folder path to scan: ").strip()
    
    # Get tolerance from user
    if len(sys.argv) > 2:
        try:
            tolerance = float(sys.argv[2])
        except ValueError:
            print(f"Error: Invalid tolerance value '{sys.argv[2]}'. Must be a number.")
            sys.exit(1)
    else:
        while True:
            try:
                tolerance_input = input("Enter tolerance percentage (default 1.0): ").strip()
                if not tolerance_input:
                    tolerance = 1.0
                    break
                tolerance = float(tolerance_input)
                if tolerance < 0 or tolerance > 100:
                    print("Tolerance must be between 0 and 100.")
                    continue
                break
            except ValueError:
                print("Please enter a valid number.")
    
    # Validate folder path
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        sys.exit(1)
    
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a directory.")
        sys.exit(1)
    
    # Convert to absolute path
    folder_path = os.path.abspath(folder_path)
    
    print(f"Solid Color Image Detector and Cleaner")
    print(f"Target folder: {folder_path}")
    print(f"Tolerance: {tolerance}% (for compression artifacts)")
    print(f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    print("\nStarting scan...\n")
    
    # Phase 1: Find and mark solid color images
    marked_files = find_and_mark_solid_images(folder_path, tolerance)
    
    # Show results
    print(f"\n{'='*50}")
    print(f"SCAN COMPLETE")
    print(f"Found {len(marked_files)} solid color images")
    print(f"{'='*50}")
    
    if not marked_files:
        print("No solid color images found. Exiting.")
        return
    
    print(f"\nMarked files:")
    for file_path in marked_files:
        print(f"  → {file_path}")
    
    # Phase 2: Wait for user confirmation
    print(f"\n{'='*50}")
    print("MANUAL REVIEW REQUIRED")
    print(f"{'='*50}")
    print("Please review the marked files in your file explorer.")
    print("Files marked with '[for deletion]' prefix will be deleted.")
    print("Make sure you're satisfied with the selection.")
    
    while True:
        response = input("\nProceed with deletion? (Y/N): ").strip().upper()
        if response in ['Y', 'YES']:
            break
        elif response in ['N', 'NO']:
            print("Deletion cancelled. Marked files remain for manual review.")
            return
        else:
            print("Please enter Y or N.")
    
    # Phase 3: Delete marked files
    delete_marked_files(folder_path)
    
    print(f"\n{'='*50}")
    print("CLEANUP COMPLETE")
    print(f"{'='*50}")
    print("All marked solid color images have been deleted.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)