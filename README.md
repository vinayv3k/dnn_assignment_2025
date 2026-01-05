StoryReasoning — Multimodal Temporal Reasoning Model (TensorFlow/Keras)
This repository implements a multimodal story reasoning system that takes a sequence of images (frames) and generates:
a caption describing the story,
optionally a reasoning explanation using a temporal LSTM over fused multimodal features.
The model is trained on the HuggingFace dataset:
daniel3303/StoryReasoning
The project includes full code for:
model.py — CNN encoder + LSTMs + multimodal fusion
train.py — training loop with 80/20 split
utils.py — dataset utilities, preprocessing
notebook.ipynb — data loading, EDA, training, evaluation, prediction
requirements.txt

🔧 Project Structure
├── model.py
├── train.py
├── utils.py
├── config.yaml
├── requirements.txt
├── README.md
└── experiments.ipynb   (optional, from notebook cells)

📦 Installation
Install dependencies:
pip install -r requirements.txt

📄 Configuration
The full training configuration is in config.yaml:

dataset:
  hf_name: "daniel3303/StoryReasoning"
  seq_len: 3
  batch_size: 16
  image_size: 128
  max_caption_len: 32
  max_reason_len: 32

model:
  image_feat_dim: 512
  text_embed_dim: 300
  text_hidden_dim: 512
  multimodal_dim: 512
  temporal_hidden_dim: 512
  text_decoder_hidden: 512
  vocab_size: 30522
  pad_token_id: 0
  bos_token_id: 101
  eos_token_id: 102
  reason_embed_dim: 256
  reason_hidden_dim: 512

training:
  lr: 1e-4
  epochs: 5
  device: "auto"
  grad_clip: 1.0
  log_interval: 50
  save_dir: "results/checkpoints"

📁 Dataset
The project automatically loads the full dataset from HuggingFace:
from datasets import load_dataset
ds = load_dataset("daniel3303/StoryReasoning")


Each example contains:
A sequence of image frames
A caption or reasoning text
The notebook converts the HuggingFace dataset into the internal format:

examples = [
  {
    'images': [PIL.Image, PIL.Image, PIL.Image],
    'caption_tokens': [101, 2052, 1023, ..., 102]
  },
  ...
]

 Model Architecture

Your model consists of:
1. CNN Encoder
ResNet50 (ImageNet-pretrained)
Outputs image_feat_dim = 512

2. Text Encoder (LSTM)
Embedding + LSTM
Generates textual feature vector

3. Multimodal Fusion
Concatenation → Linear → ReLU
Produces multimodal frame-level features

4. Temporal LSTM
LSTM over frame sequence
Produces a unified “story context”

5. Caption Decoder
LSTM that conditions on story context
Autoregressively predicts caption tokens

🧠 Training

Run training from the notebook or via CLI.
Option A — Python Notebook (Recommended)
Notebook cells include:
Loading HF dataset
Creating examples list
EDA visualizations
80/20 split using tf.data
Training loop
Evaluation & predictions

Option B — CLI
Prepare a .npz file or call from Python:
python train.py --config config.yaml --examples_pickle my_data.npz

📊 Evaluation
The notebook computes:
Token-level accuracy
Optional BLEU-style metrics
Qualitative predictions (true caption vs predicted caption)

Example visualization:
frame 0 | frame 1 | frame 2
True: 101 1002 2078 102
Pred: 101 2001 2023 102

🔮 Prediction
Run greedy decoding:
pred_tokens = model({"images_seq": batch_images}, training=False)
Use your tokenizer to convert back to text:
caption = tokenizer.decode(pred_tokens[0], skip_special_tokens=True)

💾 Checkpoints
Training saves:
results/checkpoints/
  ├── best.h5
  ├── last.h5
  ├── final_5.h5
  └── used_config.yaml


You can resume training:
model.load_weights("results/checkpoints/best.h5")

🔌 Requirements
See full file in requirements.txt:
tensorflow>=2.13
datasets>=2.17.0
transformers>=4.36.0
Pillow>=9.0
numpy>=1.23
tqdm>=4.64
pyyaml>=6.0
tensorboard>=2.13

🏁 Next Steps
You can easily extend this project:
Add reasoning decoder (same style as caption decoder)
Add beam search decoding
Use Vision Transformers instead of CNN encoder

Precompute image embeddings to speed up training

Add multiple captions or explanations per sequence
