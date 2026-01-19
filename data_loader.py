import tensorflow as tf
import os
import glob

def load_image(file_path, label, image_size=128):
    """
    1. Reads the file from disk.
    2. Decodes the JPG.
    3. Resizes it.
    4. Normalizes to [-1, 1].
    """
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [image_size, image_size])
    
    # Normalize from [0, 255] to [-1, 1] for StarGAN
    img = (img / 127.5) - 1.0
    
    return img, label

def get_dataset(dataset_path, image_size=128, batch_size=8):
    print(f"Loading custom dataset from: {dataset_path}")
    
    # Define our specific paths and the labels we want for them
    # Folder Name -> Label ID
    # 7 (Neutral) -> 0
    # 4 (Happy)   -> 1
    # 5 (Sad)     -> 2
    
    folder_map = {
        '7': 0, 
        '4': 1, 
        '5': 2
    }
    
    all_datasets = []
    
    for folder_name, label_id in folder_map.items():
        # Construct path: e.g., .../train/7/*.jpg
        # We handle both .jpg and .JPG cases just to be safe
        search_path = os.path.join(dataset_path, folder_name, "*")
        
        # Create a dataset of file paths for this specific emotion
        files_ds = tf.data.Dataset.list_files(search_path, shuffle=True)
        
        # Check if we actually found files
        if len(list(files_ds.take(1))) == 0:
            print(f"WARNING: No images found for folder '{folder_name}' in {dataset_path}")
            continue
            
        # Map file paths to (Image, Label) pairs
        labeled_ds = files_ds.map(
            lambda x: load_image(x, label_id, image_size),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        all_datasets.append(labeled_ds)
    
    if not all_datasets:
        raise ValueError("No images found! Check your path.")

    # Merge all three emotion datasets into one big dataset
    dataset = all_datasets[0]
    for ds in all_datasets[1:]:
        dataset = dataset.concatenate(ds)
    
    # Shuffle the entire combined deck
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

# --- TEST BLOCK ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # 1. PASTE YOUR EXACT PATH HERE
    # Based on your previous error log, this should be:
    path_to_train = "D:/gr1/archive/DATASET/train"
    
    try:
        ds = get_dataset(path_to_train, batch_size=4)
        print("\nSUCCESS: Dataset loaded!")
        
        # Show one batch
        for images, labels in ds.take(1):
            print(f"Labels (0=Neu, 1=Hap, 2=Sad): {labels.numpy()}")
            
            # Show the first image
            # Convert back to [0, 1] for display
            plt.imshow((images[0] + 1) / 2.0)
            plt.title(f"Label: {labels[0].numpy()}")
            plt.axis("off")
            plt.show()
            break
            
    except Exception as e:
        print(f"\nERROR: {e}")