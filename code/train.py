"""
train.py - TensorFlow/Keras training script for StoryReasoning.

Usage:
    python train.py --config config.yaml

Or import and call train_loop(config, examples) from a notebook/script.

Expect examples to be a list of dicts:
    {'images': [PIL.Image, ...] , 'caption_tokens': [int, int, ...]}

The dataset split is 80/20 train/validation.
"""
import os
import argparse
import yaml
import math
import datetime
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow import keras

from .model import build_model

# ---------------------------
# Helpers
# ---------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def preprocess_image_pil(pil_img, image_size):
    """Resize PIL image and return float32 array (H, W, C)."""
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    img = pil_img.resize((image_size, image_size), Image.BILINEAR)
    arr = np.array(img).astype(np.float32)
    # Keras ResNet preprocess expects images in BGR with mean-subtraction; model.call does keras.applications.resnet.preprocess_input
    return arr

def examples_to_numpy(examples, seq_len, image_size, max_caption_len, pad_token_id=0):
    """
    Convert list-of-examples to numpy arrays:
      images: (N, seq_len, H, W, C) float32
      captions: (N, max_caption_len) int32 (padded)
    """
    N = len(examples)
    imgs = np.zeros((N, seq_len, image_size, image_size, 3), dtype=np.float32)
    caps = np.full((N, max_caption_len), pad_token_id, dtype=np.int32)
    for i, ex in enumerate(examples):
        frames = ex['images'][:seq_len]
        if len(frames) < seq_len:
            # repeat last frame
            frames = frames + [frames[-1]] * (seq_len - len(frames))
        for s, pil_img in enumerate(frames):
            imgs[i, s] = preprocess_image_pil(pil_img, image_size)
        tok = ex.get('caption_tokens', [])[:max_caption_len]
        caps[i, :len(tok)] = np.array(tok, dtype=np.int32)
    return imgs, caps

def make_tf_dataset(images_np, captions_np, batch_size, shuffle=True):
    """
    Create tf.data.Dataset that yields (inputs_dict, targets) where:
      inputs_dict = {'images_seq': images_np, 'caption_tokens': captions_np}
      targets = captions_np (teacher forcing)
    Note: model expects images in (B, S, H, W, C)
    """
    dataset = tf.data.Dataset.from_tensor_slices((images_np, captions_np))
    if shuffle:
        dataset = dataset.shuffle(buffer_size= len(images_np) )
    # batch
    dataset = dataset.batch(batch_size, drop_remainder=False)
    # map to dict
    def _map(images, caps):
        inputs = {'images_seq': images, 'caption_tokens': caps}
        targets = caps
        return inputs, targets
    dataset = dataset.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.prefetch(tf.data.AUTOTUNE)

# ---------------------------
# Loss with mask
# ---------------------------
def masked_sparse_ce_loss(pad_token_id):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    def loss(y_true, y_pred):
        # y_true: (B, T), y_pred: (B, T, V)
        per_timestep = loss_fn(y_true, y_pred)  # (B, T)
        mask = tf.cast(tf.not_equal(y_true, pad_token_id), tf.float32)
        per_timestep = per_timestep * mask
        # average over non-pad tokens per batch element then mean over batch
        denom = tf.reduce_sum(mask, axis=1)  # (B,)
        # avoid divide by zero
        denom = tf.where(tf.equal(denom, 0.0), tf.ones_like(denom), denom)
        per_example = tf.reduce_sum(per_timestep, axis=1) / denom
        return tf.reduce_mean(per_example)
    return loss

# ---------------------------
# Training loop / entrypoint
# ---------------------------
def train_loop(config, examples, resume_ckpt=None):
    """
    config: dict parsed from config.yaml
    examples: list of dicts as described above
    resume_ckpt: optional path to restore weights
    """
    # Unpack config
    dcfg = config['dataset']
    mcfg = config['model']
    tcfg = config['training']

    seq_len = dcfg['seq_len']
    image_size = dcfg['image_size']
    max_caption_len = dcfg['max_caption_len']
    batch_size = dcfg['batch_size']
    val_ratio = 0.2  # per your request: 80/20 split

    epochs = tcfg['epochs']
    lr = tcfg['lr']
    save_dir = tcfg.get('save_dir', 'results/checkpoints')
    ensure_dir(save_dir)
    tb_logdir = os.path.join(save_dir, 'tb', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    ensure_dir(tb_logdir)

    # Convert examples to numpy arrays
    images_np, caps_np = examples_to_numpy(examples, seq_len=seq_len, image_size=image_size,
                                           max_caption_len=max_caption_len, pad_token_id=mcfg['pad_token_id'])
    N = images_np.shape[0]
    n_val = int(N * val_ratio)
    n_train = N - n_val
    if n_train <= 0:
        raise ValueError("Not enough examples for training after 80/20 split. Provide more examples.")

    # Shuffle then split
    idx = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(idx)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    train_images = images_np[train_idx]
    train_caps = caps_np[train_idx]
    val_images = images_np[val_idx]
    val_caps = caps_np[val_idx]

    train_ds = make_tf_dataset(train_images, train_caps, batch_size=batch_size, shuffle=True)
    val_ds = make_tf_dataset(val_images, val_caps, batch_size=batch_size, shuffle=False)

    # Build model
    model = build_model(config)

    # Optionally resume weights
    if resume_ckpt is not None and os.path.exists(resume_ckpt):
        print(f"Loading weights from {resume_ckpt}")
        model.load_weights(resume_ckpt)

    # Compile model
        # Compile model
    loss_fn = masked_sparse_ce_loss(mcfg['pad_token_id'])
    # Convert learning rate to float (YAML may have produced a string like "1e-4")
    raw_lr = tcfg.get('lr', 1e-3)
    try:
        lr = float(raw_lr)
    except Exception:
        raise ValueError(f"training.lr in config must be numeric or parseable as float, got: {raw_lr!r}")
    # Use clipping via optimizer (ensure grad_clip is float too)
    grad_clip = float(tcfg.get('grad_clip', 1.0))
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=grad_clip)
    model.compile(optimizer=optimizer, loss=loss_fn)


    # Callbacks: checkpoint (save best by val loss) and TensorBoard
    # Use Keras weights filename convention when saving weights only
    ckpt_best = os.path.join(save_dir, 'best.weights.h5')
    ckpt_last = os.path.join(save_dir, 'last.weights.h5')
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_best, monitor='val_loss', save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.ModelCheckpoint(ckpt_last, monitor='val_loss', save_best_only=False, save_weights_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=tb_logdir),
    ]

    # Verbosity: map log_interval to verbose (approx)
    verbose = 1 if tcfg.get('log_interval', 50) <= 50 else 2

    # Fit
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose
    )

    # Save final artifacts
    model.save_weights(os.path.join(save_dir, f'final_{epochs}.weights.h5'))
    # Also save the config for reference
    try:
        with open(os.path.join(save_dir, 'used_config.yaml'), 'w') as f:
            yaml.safe_dump(config, f)
    except Exception:
        pass

    return model, history

# ---------------------------
# CLI entrypoint
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--examples_pickle', type=str, default=None,
                        help='Optionally path to a .npz or .npickle file with preprocessed examples; otherwise you must drive train_loop from Python with `examples` list.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # If user provided a serialized examples file, try to load it (support .npz with arrays 'images' and 'caps')
    if args.examples_pickle:
        ext = os.path.splitext(args.examples_pickle)[1].lower()
        if ext == '.npz':
            data = np.load(args.examples_pickle, allow_pickle=True)
            images = data['images']
            caps = data['caps']
            # build examples list
            examples = []
            for i in range(len(images)):
                # convert numpy images back to PIL for compatibility with preprocess pipeline
                pil_frames = [Image.fromarray(images[i, s].astype('uint8')) for s in range(images.shape[1])]
                examples.append({'images': pil_frames, 'caption_tokens': list(map(int, caps[i]))})
        else:
            raise RuntimeError("Unsupported examples_pickle format. Use .npz with arrays 'images' and 'caps'.")
        model, history = train_loop(config, examples)
    else:
        raise RuntimeError("This script expects you to call train_loop(config, examples) from Python or provide --examples_pickle (.npz). See README / notebook for usage.")

if __name__ == '__main__':
    main()
