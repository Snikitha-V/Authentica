import os
from PIL import Image
from config import DATA_DIR_GENUINE, DATA_DIR_FORGED, IMG_SIZE


def preprocess_images(input_dir, output_dir, img_size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    for fname in os.listdir(input_dir):
        try:
            img_path = os.path.join(input_dir, fname)
            img = Image.open(img_path).convert("L")  # grayscale
            img = img.resize(img_size)
            img.save(os.path.join(output_dir, fname))
        except Exception as e:
            print(f"Failed to process {fname}: {e}")


if __name__ == "__main__":
    # input raw directories are in the repository root under `raw/`
    preprocess_images(os.path.join("raw", "full_org"), DATA_DIR_GENUINE, IMG_SIZE)
    preprocess_images(os.path.join("raw", "full_forg"), DATA_DIR_FORGED, IMG_SIZE)
    print("Preprocessing done!")
