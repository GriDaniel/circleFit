"""Functions for batch processing images."""
import cv2
from pathlib import Path
import argparse

from .reconstruct import reconstruct_circle_from_image

def process_images_in_folder(folder_path='test_images'):
    """Process all PNG images in the specified folder."""
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist")
        return
    
    png_files = list(folder.glob('*.png'))
    if not png_files:
        print(f"No PNG files found in '{folder_path}'")
        return
    
    output_dir = folder / 'reconstructed'
    output_dir.mkdir(exist_ok=True)
    
    print(f"Found {len(png_files)} PNG files to process in '{folder.resolve()}'")
    
    for i, png_file in enumerate(png_files, 1):
        print(f"Processing {i}/{len(png_files)}: {png_file.name}")
        
        result = reconstruct_circle_from_image(png_file)
        if result is None:
            print(f"  Failed to process {png_file.name}")
            continue
        
        output_path = output_dir / f"reconstructed_{png_file.name}"
        cv2.imwrite(str(output_path), result['image'])
        
        cx, cy = result['center']
        radius = result['radius']
        print(f"  Circle center: ({cx:.1f}, {cy:.1f}), radius: {radius:.1f}")
    
    print(f"\nResults saved in '{output_dir.resolve()}'")

def process_images_in_folder_cli():
    """Wrapper function for command-line entry point."""
    parser = argparse.ArgumentParser(description="Reconstruct circles from arc images in a folder.")
    parser.add_argument(
        '--path',
        type=str,
        default='test_images',
        help="Path to the folder containing PNG images."
    )
    args = parser.parse_args()
    process_images_in_folder(args.path)
