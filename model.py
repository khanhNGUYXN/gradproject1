import tensorflow as tf
from tensorflow.keras import layers, Model

# --- 1. Custom Helper Layers ---
class InstanceNormalization(layers.Layer):
    """
    Standard Batch Normalization normalizes across the 'Batch' (all images).
    Instance Normalization normalizes each image INDEPENDENTLY.
    This is crucial for Style Transfer/StarGAN to preserve unique details.
    """
    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

def res_block(x, filters):
    """A residual block that keeps the image content while processing style."""
    res = x
    x = layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=False)(x)
    x = InstanceNormalization()(x)
    return layers.add([res, x]) # The "Residual" connection

# --- 2. The Generator (The Artist) ---
# --- REPLACE THE 'build_generator' FUNCTION IN model.py WITH THIS ---

def build_generator(image_shape=(128, 128, 3), c_dim=3):
    """
    Diet Version: Filters are halved (64->32, 128->64, etc.)
    ResBlocks reduced from 6 to 3.
    """
    img_input = layers.Input(shape=image_shape)
    label_input = layers.Input(shape=(c_dim,))
    
    # Label processing
    label_layer = layers.RepeatVector(image_shape[0] * image_shape[1])(label_input)
    label_layer = layers.Reshape((image_shape[0], image_shape[1], c_dim))(label_layer)
    x = layers.concatenate([img_input, label_layer])

    # Down-Sampling (Lighter)
    # 64 -> 32
    x = layers.Conv2D(32, 7, strides=1, padding='same', use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = layers.ReLU()(x)

    # Down 1 (128 -> 64)
    x = layers.Conv2D(64, 4, strides=2, padding='same', use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = layers.ReLU()(x)

    # Down 2 (64 -> 32)
    x = layers.Conv2D(128, 4, strides=2, padding='same', use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = layers.ReLU()(x)

    # Bottleneck (Reduced from 6 blocks to 3)
    for _ in range(3):
        x = res_block(x, 128) # Filter size also reduced (256 -> 128)

    # Up-Sampling
    # Up 1
    x = layers.Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = layers.ReLU()(x)

    # Up 2
    x = layers.Conv2DTranspose(32, 4, strides=2, padding='same', use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = layers.ReLU()(x)

    # Output
    x = layers.Conv2D(3, 7, strides=1, padding='same', use_bias=False)(x)
    output = layers.Activation('tanh')(x)

    return Model([img_input, label_input], output, name="Generator")

# --- 3. The Discriminator (The Critic) ---
def build_discriminator(image_shape=(128, 128, 3), c_dim=3):
    """
    Input: Image
    Output 1: Real/Fake Score (Is this a real photo?)
    Output 2: Domain Classification (Does this look Happy/Sad/Neutral?)
    """
    img_input = layers.Input(shape=image_shape)

    x = layers.Conv2D(64, 4, strides=2, padding='same')(img_input) # 64x64
    x = layers.LeakyReLU(0.01)(x)

    x = layers.Conv2D(128, 4, strides=2, padding='same')(x) # 32x32
    x = layers.LeakyReLU(0.01)(x)

    x = layers.Conv2D(256, 4, strides=2, padding='same')(x) # 16x16
    x = layers.LeakyReLU(0.01)(x)

    x = layers.Conv2D(512, 4, strides=2, padding='same')(x) # 8x8
    x = layers.LeakyReLU(0.01)(x)

    x = layers.Conv2D(1024, 4, strides=2, padding='same')(x) # 4x4
    x = layers.LeakyReLU(0.01)(x)

    # Output 1: Real vs Fake (PatchGAN)
    out_src = layers.Conv2D(1, 3, strides=1, padding='same', name='output_real_fake')(x)

    # Output 2: Domain Classification (Predict the emotion)
    # Flatten the features to a single vector
    k_size = int(image_shape[0] / 32) # Calculate kernel size for 4x4
    out_cls = layers.Conv2D(c_dim, k_size, strides=1, padding='valid', name='output_label')(x)
    out_cls = layers.Flatten()(out_cls)

    return Model(img_input, [out_src, out_cls], name="Discriminator")

# --- TEST BLOCK ---
if __name__ == "__main__":
    # Quick sanity check to ensure shapes are correct
    gen = build_generator()
    disc = build_discriminator()
    
    print("Generator Output Shape:", gen.output_shape)
    # Create a random dummy image and label to test
    dummy_img = tf.random.normal([1, 128, 128, 3])
    dummy_lbl = tf.random.normal([1, 3])
    
    generated = gen([dummy_img, dummy_lbl])
    print("Successfully generated an image!")
    
    d_src, d_cls = disc(generated)
    print(f"Discriminator Real/Fake Shape: {d_src.shape}")
    print(f"Discriminator Class Shape: {d_cls.shape}")