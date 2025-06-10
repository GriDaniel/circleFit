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
    
    # Process both PNG and JPEG files for more flexibility
    image_files = list(folder.glob('*.png')) + list(folder.glob('*.jpeg')) + list(folder.glob('*.jpg'))
    if not image_files:
        print(f"No PNG or JPEG files found in '{folder_path}'")
        return
    
    output_dir = folder / 'reconstructed'
    output_dir.mkdir(exist_ok=True)
    
    print(f"Found {len(image_files)} image files to process in '{folder.resolve()}'")
    
    for i, image_file in enumerate(image_files, 1):
        # Skip any synthetic images we may have created in previous runs
        if image_file.stem.startswith('synthetic_'):
            continue
            
        print(f"\n--- Processing {i}/{len(image_files)}: {image_file.name} ---")
        
        # Pass the base folder to the reconstruction function
        result = reconstruct_circle_from_image(image_file, folder)
        
        if result is None:
            print(f"  [FAILED] Could not process {image_file.name}")
            continue
        
        # Save the final image to the 'reconstructed' sub-folder
        output_path = output_dir / f"reconstructed_{image_file.stem}.png"
        cv2.imwrite(str(output_path), result['image'])
        
        cx, cy = result['center']
        radius = result['radius']
        print(f"  [SUCCESS] Circle center: ({cx:.1f}, {cy:.1f}), radius: {radius:.1f}")
        print(f"  [SUCCESS] Saved result to: {output_path.relative_to(folder.parent)}")

    print(f"\nResults saved in '{output_dir.resolve()}'")

def process_images_in_folder_cli():
    """Wrapper function for command-line entry point."""
    parser = argparse.ArgumentParser(description="Reconstruct circles from arc images in a folder.")
    parser.add_argument(
        '--path',
        type=str,
        default='test_images',
        help="Path to the folder containing image files."
    )
    args = parser.parse_args()
    process_images_in_folder(args.path)