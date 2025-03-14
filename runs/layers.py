import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, GlobalAveragePooling2D,
    Conv2D, Activation, Multiply, Reshape, Lambda, Add, Concatenate
)

class AttentionModule(layers.Layer):
    # CBAM attention module
    def __init__(self, reduction_ratio=8, kernel_size=7, name=None, **kwargs):
        # Init attention module
        super(AttentionModule, self).__init__(name=name, **kwargs)
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size
        
    def build(self, input_shape):
        channels = input_shape[-1]
        
        # Channel attention MLP
        self.channel_dense1 = Dense(channels // self.reduction_ratio, 
                                   activation='relu',
                                   kernel_initializer='he_normal',
                                   use_bias=True,
                                   bias_initializer='zeros',
                                   kernel_regularizer=regularizers.l2(1e-5))
        
        self.channel_dense2 = Dense(channels,
                                   kernel_initializer='he_normal',
                                   use_bias=True,
                                   bias_initializer='zeros',
                                   kernel_regularizer=regularizers.l2(1e-5))
        
        # Spatial attention
        self.spatial_conv = Conv2D(1, self.kernel_size, 
                                  padding='same', 
                                  activation='sigmoid',
                                  kernel_initializer='he_normal',
                                  use_bias=False,
                                  kernel_regularizer=regularizers.l2(1e-5))
        
        super(AttentionModule, self).build(input_shape)
        
    def call(self, inputs):
        # Apply channel attention
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
        
        avg_pool = self.channel_dense1(avg_pool)
        avg_pool = self.channel_dense2(avg_pool)
        
        max_pool = self.channel_dense1(max_pool)
        max_pool = self.channel_dense2(max_pool)
        
        channel_attention = Add()([avg_pool, max_pool])
        channel_attention = Activation('sigmoid')(channel_attention)
        
        # Apply attention weights
        channel_refined = Multiply()([inputs, channel_attention])
        
        # Create spatial attention
        avg_spatial = tf.reduce_mean(channel_refined, axis=-1, keepdims=True)
        max_spatial = tf.reduce_max(channel_refined, axis=-1, keepdims=True)
        
        concat_spatial = Concatenate()([avg_spatial, max_spatial])
        spatial_attention = self.spatial_conv(concat_spatial)
        
        # Apply spatial attention
        refined = Multiply()([channel_refined, spatial_attention])
        
        return refined
    
    def get_config(self):
        config = super(AttentionModule, self).get_config()
        config.update({
            'reduction_ratio': self.reduction_ratio,
            'kernel_size': self.kernel_size
        })
        return config


class FocalLoss(tf.keras.losses.Loss):
    # For imbalanced classes
    def __init__(self, alpha=0.25, gamma=2.0, from_logits=False, name='focal_loss', **kwargs):
        super(FocalLoss, self).__init__(name=name, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
        
    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)
        
        # For binary case
        if tf.shape(y_true)[-1] == 1 or tf.shape(y_pred)[-1] == 1:
            y_true = tf.cast(y_true, tf.float32)
            # Get BCE
            bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
            
            # Focal weights
            p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
            alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
            modulating_factor = tf.pow(1.0 - p_t, self.gamma)
            
            # Apply weights
            loss = alpha_factor * modulating_factor * bce
            return tf.reduce_mean(loss)
        else:
            # For multi-class
            y_true = tf.cast(y_true, tf.float32)
            # Get CCE
            ce = tf.keras.backend.categorical_crossentropy(y_true, y_pred)
            
            # Focal weights
            p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
            modulating_factor = tf.pow(1.0 - p_t, self.gamma)
            
            # Alpha weights
            alpha_tensor = tf.ones_like(y_true) * self.alpha
            alpha_weight = tf.reduce_sum(alpha_tensor * y_true, axis=-1)
            
            # Apply weights
            loss = alpha_weight * modulating_factor * ce
            return tf.reduce_mean(loss)
    
    def get_config(self):
        config = super(FocalLoss, self).get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma,
            'from_logits': self.from_logits
        })
        return config


def apply_mixup(x, y, alpha=0.2):
    # Mix pairs of samples
    batch_size = tf.shape(x)[0]
    
    # Get mixing weights
    beta = tf.random.uniform([batch_size], 0, 1)
    
    # Get random indices
    indices = tf.random.shuffle(tf.range(batch_size))
    
    # Get permutations
    x_perm = tf.gather(x, indices)
    y_perm = tf.gather(y, indices)
    
    # Get mix ratio
    if alpha > 0:
        beta = tf.maximum(beta, 1-beta)
    else:
        beta = tf.ones_like(beta)
    
    # Reshape for broadcast
    beta_x = tf.reshape(beta, [batch_size, 1, 1, 1])
    beta_y = tf.reshape(beta, [batch_size, 1])
    
    # Mix data and labels
    mixed_x = beta_x * x + (1 - beta_x) * x_perm
    mixed_y = beta_y * y + (1 - beta_y) * y_perm
    
    return mixed_x, mixed_y


def apply_cutmix(x, y, alpha=1.0):
    # Cut and mix regions
    batch_size = tf.shape(x)[0]
    image_height = tf.shape(x)[1]
    image_width = tf.shape(x)[2]
    
    # Get random indices
    indices = tf.random.shuffle(tf.range(batch_size))
    
    # Get permutations
    x_perm = tf.gather(x, indices)
    y_perm = tf.gather(y, indices)
    
    # Get mixing ratio
    if alpha > 0:
        lam = tf.random.uniform([], 0, 1)
    else:
        lam = 0.5
    
    # Get box dimensions
    cut_ratio = tf.sqrt(1.0 - lam)
    cut_h = tf.cast(tf.cast(image_height, tf.float32) * cut_ratio, tf.int32)
    cut_w = tf.cast(tf.cast(image_width, tf.float32) * cut_ratio, tf.int32)
    
    # Get box center
    center_x = tf.random.uniform([], 0, image_width, dtype=tf.int32)
    center_y = tf.random.uniform([], 0, image_height, dtype=tf.int32)
    
    # Get box coordinates
    x1 = tf.maximum(0, center_x - cut_w // 2)
    y1 = tf.maximum(0, center_y - cut_h // 2)
    x2 = tf.minimum(image_width, center_x + cut_w // 2)
    y2 = tf.minimum(image_height, center_y + cut_h // 2)
    
    # Create float versions
    x1_f = tf.cast(x1, tf.float32)
    y1_f = tf.cast(y1, tf.float32)
    x2_f = tf.cast(x2, tf.float32)
    y2_f = tf.cast(y2, tf.float32)
    
    # Calculate actual area
    actual_area = (x2_f - x1_f) * (y2_f - y1_f)
    total_area = tf.cast(image_height * image_width, tf.float32)
    actual_lam = 1.0 - actual_area / total_area
    
    # Create cut mask
    mask = tf.cast(
        tf.logical_and(
            tf.logical_and(
                tf.range(image_height)[:, tf.newaxis] >= y1, 
                tf.range(image_height)[:, tf.newaxis] < y2
            ),
            tf.logical_and(
                tf.range(image_width)[tf.newaxis, :] >= x1,
                tf.range(image_width)[tf.newaxis, :] < x2
            )
        ),
        tf.float32
    )
    mask = mask[:, :, tf.newaxis, tf.newaxis]
    
    # Apply cut mask
    mixed_x = x * (1.0 - mask) + x_perm * mask
    mixed_y = actual_lam * y + (1.0 - actual_lam) * y_perm
    
    return mixed_x, mixed_y