"""
utils.py

TensorFlow-focused utilities for the StoryReasoning project.

Main features:
- set_seed(seed): deterministically set random seeds for reproducibility
- preprocess_image_pil / preprocess_image_np: put images into shape/resnet preprocess
- examples_to_numpy: convert examples list -> numpy arrays for images and captions
- build_tf_datasets: build tf.data.Dataset train/val with 80/20 split (configurable)
- save_weights / load_weights: simple wrappers for model weight persistence
- make_synthetic_examples: quick toy generator used in the notebook
"""

import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf

# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------
# Image preprocessing helpers
# ---------------------------
def preprocess_image_pil(pil_img: Image.Image, image_size: int):
    """Resize a PIL.Image to (image_size, image_size) and return numpy float32 array.
    The returned array is suitable to pass into keras.applications.resnet.preprocess_input.
    """
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    img = pil_img.resize((image_size, image_size), Image.BILINEAR)
    arr = np.array(img).astype(np.float32)
    return arr  # (H, W, C) float32


def preprocess_image_np(arr: np.ndarray, image_size: int):
    """Resize and convert a numpy image (H,W,C, uint8/float) to float32 (H,W,C)."""
    if arr.dtype != np.uint8:
        # assume float in 0..1 -> scale to 0..255
        arr = (arr * 255.0).astype(np.uint8)
    pil = Image.fromarray(arr)
    return preprocess_image_pil(pil, image_size)


# ---------------------------
# Convert examples -> numpy arrays
# ---------------------------
def examples_to_numpy(examples, seq_len: int, image_size: int, max_caption_len: int, pad_token_id: int = 0):
    """
    Convert list of examples to numpy arrays:
      - examples: list of dicts {'images': [PIL.Image,...], 'caption_tokens': [int,...]}
      - returns:
          images_np: (N, seq_len, H, W, C) float32
          caps_np:   (N, max_caption_len) int32 (padded with pad_token_id)
    Notes:
      - If an example has less than seq_len frames, the last frame is repeated.
      - If more, only the first seq_len frames are used.
    """
    N = len(examples)
    imgs = np.zeros((N, seq_len, image_size, image_size, 3), dtype=np.float32)
    caps = np.full((N, max_caption_len), pad_token_id, dtype=np.int32)
    for i, ex in enumerate(examples):
        frames = ex.get('images', [])[:seq_len]
        if len(frames) == 0:
            raise ValueError(f"Example {i} has no images.")
        if len(frames) < seq_len:
            frames = frames + [frames[-1]] * (seq_len - len(frames))
        for s, pil_img in enumerate(frames):
            imgs[i, s] = preprocess_image_pil(pil_img, image_size)
        tok = ex.get('caption_tokens', [])[:max_caption_len]
        if len(tok) > 0:
            caps[i, :len(tok)] = np.array(tok, dtype=np.int32)
    return imgs, caps


# ---------------------------
# tf.data pipeline builder
# ---------------------------
def make_tf_dataset_from_arrays(images_np, caps_np, batch_size: int, shuffle: bool = True):
    """
    Turn numpy arrays into a tf.data.Dataset yielding (inputs_dict, targets)
    where inputs_dict = {'images_seq': images, 'caption_tokens': caps}
    and targets = caps (teacher forcing).
    """
    dataset = tf.data.Dataset.from_tensor_slices((images_np, caps_np))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=max(1024, len(images_np)))
    dataset = dataset.batch(batch_size)
    def _map(images, caps):
        # Note: images expected in (B, S, H, W, C)
        # Keras model's call will run preprocess_input internally in model.py; if you prefer, apply it here.
        inputs = {'images_seq': tf.cast(images, tf.float32), 'caption_tokens': tf.cast(caps, tf.int32)}
        targets = caps
        return inputs, targets
    dataset = dataset.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.prefetch(tf.data.AUTOTUNE)


def build_tf_datasets_from_examples(examples, config, val_ratio: float = 0.2, seed: int = 42):
    """
    Build train/val tf.data.Dataset objects from raw `examples` list.
    - uses 80/20 split by default.
    - config is the parsed config.yaml dict (expects dataset & model keys).
    Returns: train_ds, val_ds, (train_count, val_count)
    """
    set_seed(seed)
    dcfg = config['dataset']
    mcfg = config['model']

    seq_len = dcfg['seq_len']
    image_size = dcfg['image_size']
    max_caption_len = dcfg['max_caption_len']
    batch_size = dcfg['batch_size']

    images_np, caps_np = examples_to_numpy(examples, seq_len=seq_len,
                                           image_size=image_size,
                                           max_caption_len=max_caption_len,
                                           pad_token_id=mcfg['pad_token_id'])
    N = images_np.shape[0]
    if N == 0:
        raise ValueError("No examples provided.")
    # Shuffle indices then split
    idx = np.arange(N)
    np.random.seed(seed)
    np.random.shuffle(idx)
    n_val = int(N * val_ratio)
    n_train = N - n_val
    if n_train <= 0:
        raise ValueError("Not enough examples for training after split.")
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]
    train_images = images_np[train_idx]
    train_caps = caps_np[train_idx]
    val_images = images_np[val_idx]
    val_caps = caps_np[val_idx]

    train_ds = make_tf_dataset_from_arrays(train_images, train_caps, batch_size=batch_size, shuffle=True)
    val_ds = make_tf_dataset_from_arrays(val_images, val_caps, batch_size=batch_size, shuffle=False)

    return train_ds, val_ds, (n_train, n_val)


# ---------------------------
# Checkpoint helpers
# ---------------------------
def save_weights(model: tf.keras.Model, path: str):
    """Save model weights to given path (creates directories)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save_weights(path)


def load_weights(model: tf.keras.Model, path: str):
    """Load model weights from given path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model.load_weights(path)


# ---------------------------
# Small synthetic dataset generator (useful for quick tests)
# ---------------------------
def make_synthetic_examples(N: int, seq_len: int, image_size: int, max_caption_len: int,
                            vocab_low: int = 999, vocab_high: int = 2000):
    """
    Generate N toy examples:
      - images are random noise PIL images
      - captions are short random token id lists (start=101, random tokens, end=102)
    Useful for testing the pipeline quickly (not for real training).
    """
    examples = []
    for i in range(N):
        frames = []
        for s in range(seq_len):
            arr = (np.random.rand(image_size, image_size, 3) * 255).astype('uint8')
            frames.append(Image.fromarray(arr))
        # random length 3..(max_caption_len-2)
        cap_len = min(max_caption_len - 2, max(3, np.random.randint(3, 8)))
        cap = [101] + [int(np.random.randint(vocab_low, vocab_high)) for _ in range(cap_len)] + [102]
        examples.append({'images': frames, 'caption_tokens': cap})
    return examples


# ---------------------------
# Small convenience runner used by notebooks
# ---------------------------
if __name__ == "__main__":
    # Quick self-test: make small synthetic dataset and build ds
    from pprint import pprint
    set_seed(42)
    example_cfg = {
        'dataset': {
            'seq_len': 3,
            'image_size': 128,
            'max_caption_len': 32,
            'batch_size': 8
        },
        'model': {
            'pad_token_id': 0
        }
    }
    ex = make_synthetic_examples(N=32, seq_len=3, image_size=128, max_caption_len=16)
    train_ds, val_ds, counts = build_tf_datasets_from_examples(ex, {**example_cfg, 'model': example_cfg['model']})
    print("Built datasets. Train/Val sizes:", counts)
    for batch in train_ds.take(1):
        inputs, targets = batch
        pprint({k: v.shape for k, v in inputs.items()})
        print("targets shape:", targets.shape)
