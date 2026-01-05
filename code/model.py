"""
model.py

TensorFlow / Keras implementation of StoryReasoning model based on config.yaml.

Components:
- CNNEncoder: ResNet50 (no top) + Dense -> image_feat_dim
- TextEncoder: Embedding + LSTM -> text_hidden_dim
- ReasonEncoder: optional (different embedding dim)
- MultimodalFusion: Dense(ReLU) on concatenated image+text features
- TemporalReasonLSTM: LSTM over sequence of multimodal vectors -> temporal_hidden_dim
- CaptionDecoder: Embedding + LSTM that conditions on temporal context and outputs logits over vocab.

Usage:
    model = build_model(config)
    # For training pass dict: {"images_seq": images, "caption_tokens": caps}
    # For inference pass dict: {"images_seq": images} and model will return greedy token ids
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore


class CNNEncoder(keras.layers.Layer):
    def __init__(self, image_feat_dim=512, pretrained=True, name="cnn_encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        # Use ResNet50 as backbone, without top, average pooling
        self.backbone = keras.applications.ResNet50(include_top=False, weights='imagenet' if pretrained else None,
                                                    pooling='avg')
        self.proj = layers.Dense(image_feat_dim, activation=None, name="img_proj")

    def call(self, images, training=False):
        # images: (B, C, H, W) or (B, H, W, C) depending on preprocess
        # Keras expects (B, H, W, C) - ensure input is in that format before calling
        feats = self.backbone(images, training=training)  # (B, feat)
        feats = self.proj(feats)  # (B, image_feat_dim)
        return feats


class TextEncoder(keras.layers.Layer):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_token_id=0, name="text_encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.embed = layers.Embedding(vocab_size, embed_dim, mask_zero=(pad_token_id == 0), name="text_embed")
        self.lstm = layers.LSTM(hidden_dim, return_state=True, name="text_lstm")

    def call(self, token_ids, training=False):
        """
        token_ids: (B, T)
        returns: (B, hidden_dim) last hidden state
        """
        x = self.embed(token_ids)  # (B, T, emb)
        # Using Keras LSTM which handles mask automatically when mask_zero True
        output, h, c = self.lstm(x, training=training)
        return h


class ReasonEncoder(keras.layers.Layer):
    def __init__(self, vocab_size, reason_embed_dim, reason_hidden_dim, pad_token_id=0, name="reason_encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.embed = layers.Embedding(vocab_size, reason_embed_dim, mask_zero=(pad_token_id == 0),
                                      name="reason_embed")
        self.lstm = layers.LSTM(reason_hidden_dim, return_state=True, name="reason_lstm")

    def call(self, token_ids, training=False):
        x = self.embed(token_ids)
        out, h, c = self.lstm(x, training=training)
        return h


class MultimodalFusion(keras.layers.Layer):
    def __init__(self, image_dim, text_dim, multimodal_dim, name="fusion", **kwargs):
        super().__init__(name=name, **kwargs)
        self.fc = layers.Dense(multimodal_dim, activation="relu", name="fusion_fc")

    def call(self, image_feat, text_feat):
        # image_feat: (B, image_dim), text_feat: (B, text_dim)
        x = tf.concat([image_feat, text_feat], axis=-1)
        return self.fc(x)


class TemporalReasonLSTM(keras.layers.Layer):
    def __init__(self, multimodal_dim, temporal_hidden_dim, name="temporal_lstm", **kwargs):
        super().__init__(name=name, **kwargs)
        self.lstm = layers.LSTM(temporal_hidden_dim, return_state=True, name="temporal_lstm_layer")

    def call(self, multimodal_seq, training=False):
        """
        multimodal_seq: (B, seq_len, multimodal_dim)
        returns: (B, temporal_hidden_dim) last hidden state
        """
        out, h, c = self.lstm(multimodal_seq, training=training)
        return h


class CaptionDecoder(keras.layers.Layer):
    def __init__(self, vocab_size, embed_dim, hidden_dim, multimodal_context_dim,
                 pad_token_id=0, bos_token_id=101, name="caption_decoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.vocab_size = vocab_size
        self.embed = layers.Embedding(vocab_size, embed_dim, mask_zero=(pad_token_id == 0), name="dec_embed")
        # The decoder LSTM input will be [token_emb ; context] concatenated
        self.lstm = layers.LSTM(hidden_dim, return_sequences=True, return_state=True, name="dec_lstm")
        self.out = layers.Dense(vocab_size, activation=None, name="dec_logits")
        self.multimodal_context_dim = multimodal_context_dim
        self.bos_token_id = bos_token_id

    def call(self, context, token_ids=None, max_len=32, training=False):
        """
        If token_ids provided -> teacher forcing logits: returns (B, T, V)
        Else -> greedy decode returns token ids (B, max_len)
        context: (B, multimodal_context_dim)
        token_ids: (B, T)
        """
        B = tf.shape(context)[0]

        if token_ids is not None:
            # Teacher forcing path
            emb = self.embed(token_ids)  # (B, T, emb)
            # expand context to (B, T, multimodal_context_dim)
            ctx = tf.expand_dims(context, axis=1)
            ctx = tf.tile(ctx, [1, tf.shape(emb)[1], 1])
            dec_inp = tf.concat([emb, ctx], axis=-1)  # (B, T, emb+ctx)
            seq_out, h, c = self.lstm(dec_inp, training=training)
            logits = self.out(seq_out)  # (B, T, V)
            return logits
        else:
            # Greedy decode (eager loop)
            # initialize with BOS
            outputs = []
            token = tf.fill([B, 1], tf.cast(self.bos_token_id, tf.int32))  # (B,1)
            h, c = None, None
            for t in range(max_len):
                emb = self.embed(token)  # (B,1,emb)
                emb = tf.squeeze(emb, axis=1)  # (B, emb)
                inp = tf.concat([emb, context], axis=-1)  # (B, emb+ctx)
                inp = tf.expand_dims(inp, axis=1)  # (B,1, input_dim)
                if h is None:
                    seq_out, h, c = self.lstm(inp)  # first step
                else:
                    seq_out, h, c = self.lstm(inp, initial_state=[h, c])
                logit = self.out(tf.squeeze(seq_out, axis=1))  # (B, V)
                token = tf.expand_dims(tf.argmax(logit, axis=-1, output_type=tf.int32), axis=1)  # (B,1)
                outputs.append(token)
            outputs = tf.concat(outputs, axis=1)  # (B, max_len)
            return outputs


class StoryReasoningModel(keras.Model):
    def __init__(self, config, name="story_reasoning_model", **kwargs):
        super().__init__(name=name, **kwargs)
        mcfg = config['model']
        dcfg = config['dataset']
        self.config = config

        # Encoder/decoder components
        self.image_encoder = CNNEncoder(image_feat_dim=mcfg['image_feat_dim'], pretrained=True)
        self.text_encoder = TextEncoder(vocab_size=mcfg['vocab_size'],
                                        embed_dim=mcfg['text_embed_dim'],
                                        hidden_dim=mcfg['text_hidden_dim'],
                                        pad_token_id=mcfg['pad_token_id'])
        # Reason encoder (optional different dims)
        self.reason_encoder = ReasonEncoder(vocab_size=mcfg['vocab_size'],
                                            reason_embed_dim=mcfg.get('reason_embed_dim', mcfg['text_embed_dim']),
                                            reason_hidden_dim=mcfg.get('reason_hidden_dim', mcfg['text_hidden_dim']),
                                            pad_token_id=mcfg['pad_token_id'])
        self.fusion = MultimodalFusion(image_dim=mcfg['image_feat_dim'],
                                       text_dim=mcfg['text_hidden_dim'],
                                       multimodal_dim=mcfg['multimodal_dim'])
        self.temporal = TemporalReasonLSTM(multimodal_dim=mcfg['multimodal_dim'],
                                          temporal_hidden_dim=mcfg['temporal_hidden_dim'])
        self.caption_decoder = CaptionDecoder(vocab_size=mcfg['vocab_size'],
                                              embed_dim=mcfg['text_embed_dim'],
                                              hidden_dim=mcfg['text_decoder_hidden'],
                                              multimodal_context_dim=mcfg['temporal_hidden_dim'],
                                              pad_token_id=mcfg['pad_token_id'],
                                              bos_token_id=mcfg['bos_token_id'])

    def call(self, inputs, training=False):
        """
        inputs: dict with keys:
            - 'images_seq': Tensor (B, seq_len, H, W, C)  [Keras default channel-last]
            - optionally 'caption_tokens': Tensor (B, T)
        returns:
            - if caption_tokens given: logits (B, T, V)
            - else: predicted token ids (B, max_reason_len)
        """
        images_seq = inputs['images_seq']  # (B, S, H, W, C)
        caption_tokens = inputs.get('caption_tokens', None)

        B = tf.shape(images_seq)[0]
        S = tf.shape(images_seq)[1]

        # Flatten frames to encode per-image then reshape
        images_flat = tf.reshape(images_seq, (-1, tf.shape(images_seq)[2], tf.shape(images_seq)[3], tf.shape(images_seq)[4]))
        # Preprocess images for ResNet (assumes images in 0..255 uint8 or floats); convert to float32 and apply keras preprocess
        images_flat = tf.cast(images_flat, tf.float32)
        images_flat = keras.applications.resnet.preprocess_input(images_flat)
        img_feats_flat = self.image_encoder(images_flat, training=training)  # (B*S, image_feat_dim)
        img_feats = tf.reshape(img_feats_flat, (B, S, -1))  # (B, S, image_feat_dim)

        # Here we assume no per-frame text; use zeros for text_feat (can be replaced if per-frame captions exist)
        text_feat = tf.zeros((B, self.config['model']['text_hidden_dim']), dtype=tf.float32)

        # Build multimodal seq
        mm_list = []
        for s in range(int(self.config['dataset']['seq_len'])):
            # Taking img_feats[:, s, :] (note: using static seq_len from config for python loop stability)
            mm = self.fusion(img_feats[:, s, :], text_feat)
            mm_list.append(tf.expand_dims(mm, axis=1))
        multimodal_seq = tf.concat(mm_list, axis=1)  # (B, S, multimodal_dim)

        # Temporal context vector
        seq_context = self.temporal(multimodal_seq, training=training)  # (B, temporal_hidden_dim)

        if caption_tokens is not None:
            logits = self.caption_decoder(seq_context, token_ids=caption_tokens, max_len=self.config['dataset']['max_caption_len'],
                                          training=training)
            return logits
        else:
            preds = self.caption_decoder(seq_context, token_ids=None, max_len=self.config['dataset']['max_reason_len'],
                                         training=training)
            return preds


def build_model(config):
    """
    Utility to create and build the model. After calling, you may compile the model
    with optimizer/loss or use custom training loops. We run one dummy build pass
    so shapes are known.
    """
    model = StoryReasoningModel(config)
    # Dummy build: create dummy inputs with shapes from config
    seq_len = config['dataset']['seq_len']
    img_h = config['dataset'].get('image_size', 128)
    img_w = img_h
    channels = 3
    batch = 2
    dummy_images = tf.zeros((batch, seq_len, img_h, img_w, channels), dtype=tf.float32)
    # call once to build weights
    _ = model({'images_seq': dummy_images}, training=False)
    return model
