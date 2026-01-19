import tensorflow as tf
import time
import os
import matplotlib.pyplot as plt
from data_loader import get_dataset
from model import build_generator, build_discriminator

# --- GPU MEMORY FIX (Crucial for GTX 960M) ---
# This prevents TensorFlow from hoarding all VRAM and crashing
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Success: GPU Memory Growth Enabled")
    except RuntimeError as e:
        print(e)

# --- CONFIGURATION ---
# Path to your dataset (Same as in data_loader.py)
DATASET_PATH = "D:/gr1/archive/DATASET/train" 

BATCH_SIZE = 1       # <--- CHANGED TO 1 (Safe Mode)
EPOCHS = 50          
LR = 1e-4            
C_DIM = 3            

# Create folders to save results
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("samples", exist_ok=True)

# ... (The rest of the file stays exactly the same) ...

# --- 1. SETUP ---
# Load Data
dataset = get_dataset(DATASET_PATH, batch_size=BATCH_SIZE)

# Build Models
generator = build_generator(c_dim=C_DIM)
discriminator = build_discriminator(c_dim=C_DIM)

# Optimizers
g_optimizer = tf.keras.optimizers.Adam(learning_rate=LR, beta_1=0.5, beta_2=0.999)
d_optimizer = tf.keras.optimizers.Adam(learning_rate=LR, beta_1=0.5, beta_2=0.999)

# Loss Functions
l1_loss = tf.keras.losses.MeanAbsoluteError() # For reconstruction (Identity)
mse_loss = tf.keras.losses.MeanSquaredError() # For realism (LSGAN)
cce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True) # For emotion labels

# --- 2. THE TRAINING STEP (Runs on GPU) ---
@tf.function
def train_step(real_img, real_label_int):
    # Convert integer labels (0, 1, 2) to One-Hot ([1,0,0], ...)
    real_label = tf.one_hot(real_label_int, depth=C_DIM)
    
    # Generate a random target emotion for training
    # (e.g., Change this Neutral face to Happy)
    # We create random indices [0, 1, 2] and convert to one-hot
    rand_idx = tf.random.uniform(shape=[BATCH_SIZE], minval=0, maxval=C_DIM, dtype=tf.int32)
    target_label = tf.one_hot(rand_idx, depth=C_DIM)

    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        # --- GENERATOR ACTION ---
        # 1. Fake Image: Translate Real Image -> Target Emotion
        fake_img = generator([real_img, target_label], training=True)
        
        # 2. Reconstruction: Translate Fake Image -> Back to Original Emotion
        rec_img = generator([fake_img, real_label], training=True)
        
        # --- DISCRIMINATOR ACTION ---
        # Check Real Image
        d_out_real, d_cls_real = discriminator(real_img, training=True)
        # Check Fake Image
        d_out_fake, d_cls_fake = discriminator(fake_img, training=True)
        
        # --- CALCULATE LOSSES ---
        
        # 1. Discriminator Loss (Teach it to spot fakes & classify real emotions)
        d_loss_real = mse_loss(tf.ones_like(d_out_real), d_out_real) # Real should be 1
        d_loss_fake = mse_loss(tf.zeros_like(d_out_fake), d_out_fake) # Fake should be 0
        d_loss_cls = cce_loss(real_label, d_cls_real) # Should classify real face correctly
        
        d_loss = d_loss_real + d_loss_fake + d_loss_cls
        
        # 2. Generator Loss (Teach it to fool D, change emotion, and keep identity)
        g_loss_adv = mse_loss(tf.ones_like(d_out_fake), d_out_fake) # Fake should be 1 (Fool D)
        g_loss_cls = cce_loss(target_label, d_cls_fake) # Fake image should look like Target Emotion
        g_loss_rec = l1_loss(real_img, rec_img) # Rec image should look like Real Image
        
        # Weighted Sum (Weights taken from standard StarGAN paper)
        g_loss = g_loss_adv + (1.0 * g_loss_cls) + (10.0 * g_loss_rec)

    # --- UPDATE GRADIENTS ---
    g_gradients = g_tape.gradient(g_loss, generator.trainable_variables)
    d_gradients = d_tape.gradient(d_loss, discriminator.trainable_variables)
    
    g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
    d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))
    
    return g_loss, d_loss

# --- 3. HELPER: SAVE SAMPLE IMAGES ---
def generate_and_save_images(model, test_input, epoch):
    # Target: Happy (Label 1)
    target_label = tf.one_hot([1]*BATCH_SIZE, depth=C_DIM) 
    
    prediction = model([test_input, target_label], training=False)

    plt.figure(figsize=(8, 4))
    
    # Show Input (Real)
    plt.subplot(1, 2, 1)
    plt.title("Input (Neutral/Real)")
    plt.imshow((test_input[0] + 1) / 2.0)
    plt.axis('off')

    # Show Output (Fake Happy)
    plt.subplot(1, 2, 2)
    plt.title("Generated (Happy)")
    plt.imshow((prediction[0] + 1) / 2.0)
    plt.axis('off')

    plt.savefig(f"samples/epoch_{epoch:03d}.png")
    plt.close()

# --- REPLACE THE ENTIRE MAIN LOOP AT THE BOTTOM OF train.py WITH THIS ---

print("Starting Training...")
print(f"Check the 'samples' folder to see progress!")

# Grab one batch for visualization
sample_batch, _ = next(iter(dataset))

# Benchmark tracking
total_steps_per_epoch = 12271  # Approx size of RAF-DB training set
print_interval = 20            # Print every 20 steps (More frequent updates)

for epoch in range(EPOCHS):
    epoch_start_time = time.time()
    step_start_time = time.time()
    step_count = 0
    
    # Iterate through the dataset
    for image_batch, label_batch in dataset:
        g_loss, d_loss = train_step(image_batch, label_batch)
        step_count += 1
        
        # PRINT PROGRESS EVERY 20 STEPS
        if step_count % print_interval == 0:
            # Calculate speed
            elapsed = time.time() - step_start_time
            steps_per_sec = print_interval / elapsed
            
            # Estimate time remaining in this epoch
            steps_left = total_steps_per_epoch - step_count
            mins_left = (steps_left / steps_per_sec) / 60
            
            print(f"Epoch {epoch+1} | Step {step_count} | G Loss: {g_loss:.4f} | D Loss: {d_loss:.4f} | Speed: {steps_per_sec:.2f} img/s | ~{mins_left:.1f} mins left in epoch")
            
            # Reset timer for next batch
            step_start_time = time.time()

    # End of Epoch actions
    generate_and_save_images(generator, sample_batch, epoch)
    
    # Save model weights every 5 epochs
    if (epoch + 1) % 5 == 0:
        generator.save_weights(f"checkpoints/gen_{epoch+1}.h5")
        print(f"Saved checkpoint for epoch {epoch+1}")

    print(f'>>> Epoch {epoch+1} FINISHED | Total Time: {(time.time()-epoch_start_time)/60:.1f} min')