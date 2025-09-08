import subprocess
import os

def process_images():
    # Paths!
    txt_path = "/home/ml3/Desktop/Thesis/.venv/Medieval/Text-Line-Segmentation-Method-for-Medieval-Manuscripts-master/src/line_segmentation/ICDAR2012_lines.txt"
    images_dir = "/home/ml3/Desktop/Thesis/.venv/NewImplementation/Images"
    
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    # Process each image
    for i in range(100):            # 100 images (027 to 126)
        image_base = f"{i+27:03d}"  # 027, 028, ..., 126
        
        for variant in range(1, 5):  # variants 1-4
            image_name = f"{image_base}_{variant}.tif"
            image_path = os.path.join(images_dir, image_name)
            
            if not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found, skipping...")
                continue
            
            # Getting the expected lines from the text file.
            # Each line corresponds to one image (100 lines total).
            # Each line has 4 numbers for the 4 variants.
            line_data = lines[i].strip().split()
            if len(line_data) != 4:
                print(f"Warning: Invalid format in line {i+1}, skipping {image_name}")
                continue

            # Adding 1 to capture the bottom of the last line.
            expected_lines = int(line_data[variant-1]) + 1  
            
            # Building the command!
            cmd = [
                "python", "-m", "src.line_segmentation.line_segmentation",
                "--input-path", image_path,
                "--expected_lines", str(expected_lines)
            ]
            
            print(f"Processing {image_name} with expected_lines={expected_lines}")
            try:
                result = subprocess.run(cmd, check=True, cwd="/home/ml3/Desktop/Thesis/.venv/Medieval/Text-Line-Segmentation-Method-for-Medieval-Manuscripts-master")
                print(f"Successfully processed {image_name}")
            except subprocess.CalledProcessError as e:
                print(f"Error processing {image_name}: {e}")
            except Exception as e:
                print(f"Unexpected error with {image_name}: {e}")

if __name__ == "__main__":
    process_images()
