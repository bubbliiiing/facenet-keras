import math
from functools import partial

import keras.backend as K
import tensorflow as tf


def triplet_loss(alpha = 0.2, batch_size = 32):
    def _triplet_loss(y_true, y_pred):
        anchor, positive, negative = y_pred[:batch_size], y_pred[batch_size:int(2 * batch_size)], y_pred[-batch_size:]

        pos_dist    = K.sqrt(K.sum(K.square(anchor - positive), axis=-1))
        neg_dist    = K.sqrt(K.sum(K.square(anchor - negative), axis=-1))

        basic_loss  = pos_dist - neg_dist + alpha
        
        idxs        = tf.where(basic_loss > 0)
        select_loss = tf.gather_nd(basic_loss, idxs)

        loss        = K.sum(K.maximum(basic_loss, 0)) / tf.cast(tf.maximum(1, tf.shape(select_loss)[0]), tf.float32)
        return loss
    return _triplet_loss

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.1, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.3, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2
            ) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0
                + math.cos(
                    math.pi
                    * (iters - warmup_total_iters)
                    / (total_iters - warmup_total_iters - no_aug_iter)
                )
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

