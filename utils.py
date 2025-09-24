import numpy as np
import tensorflow as tf
import io
import matplotlib.pyplot as plt

class ImageLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, data_generator_trn, data_generator_tst, num_images=3):
        super().__init__()
        self.file_writer = tf.summary.create_file_writer(log_dir + '/images')
        self.data_gen_trn = data_generator_trn
        self.data_gen_tst = data_generator_tst
        self.num_images = num_images

    def on_epoch_end(self, epoch, logs=None):
        # Only after first epoch
        if epoch == 0:
            # Get a batch
            x_batch_trn, y_batch_trn = next(iter(self.data_gen_trn))
            x_batch_tst, y_batch_tst = next(iter(self.data_gen_tst))
            #x_batch /= 255
            # Run prediction
            preds_trn = self.model.predict(x_batch_trn[:self.num_images])
            preds_tst = self.model.predict(x_batch_tst[:self.num_images])

            # Convert to images for TensorBoard
            with self.file_writer.as_default():
                for i in range(self.num_images):
                    fig, ax = plt.subplots(1, 2, figsize=(6,3))
                    #ax.imshow(x_batch[i].astype("uint8"))
                    ax[0].imshow(x_batch_trn[i]/255)
                    ax[0].set_title(f"gt: {y_batch_trn[i]} pred: {preds_trn[i]}")
                    ax[0].axis("off")
                    #TST
                    ax[1].imshow(x_batch_tst[i]/255)
                    ax[1].set_title(f"gt tst: {y_batch_tst[i]} pred: {preds_tst[i]}")
                    ax[1].axis("off")
                    
                    # Convert figure to image
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    plt.close(fig)
                    buf.seek(0)
                    image = tf.image.decode_png(buf.getvalue(), channels=4)
                    image = tf.expand_dims(image, 0)  # add batch dimension
                    
                    # Write to TensorBoard
                    tf.summary.image(f"Sample_{i}", image, step=epoch)

def _add_glare(x, p=0.9, max_alpha=0.6):
    """Add a simple lens-glare spot with a radial falloff.
    Expects x in [0,255] float or uint8, shape (H,W,C) with C=1 or 3."""
    if np.random.rand() > p:
        return x

    x = x.astype(np.float32, copy=True)
    h, w = x.shape[:2]
    # random glare center (often near top/edge looks natural)
    cx = np.random.uniform(-0.2 * w, 1.2 * w)
    cy = np.random.uniform(-0.2 * h, 0.7 * h)
    # random radius
    R = np.random.uniform(0.2, 0.6) * min(h, w)

    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = np.clip(1.0 - dist / R, 0.0, 1.0)
    mask = mask ** 2  # smoother falloff

    # intensity and (slightly warm) color tint
    alpha = np.random.uniform(0.25, max_alpha)
    if x.shape[-1] == 3:
        tint = np.array([1.0, 0.95, 0.85], dtype=np.float32)  # warm glare
        glare = (255.0 * alpha * mask)[..., None] * tint[None, None, :]
    else:
        glare = (255.0 * alpha * mask)[..., None]

    x += glare
    np.clip(x, 0, 255, out=x)
    return x

def random_contrast_np(x, lower=0.8, upper=1.25):
    """Contrast jitter (NumPy). x expected in [0,255], shape (H,W,C)."""
    x = x.astype(np.float32, copy=True)
    factor = np.random.uniform(lower, upper)
    mean = np.mean(x, axis=(0, 1), keepdims=True)
    x = (x - mean) * factor + mean
    np.clip(x, 0, 255, out=x)
    return x

def glare_then_preprocess(x):
    x = _add_glare(x, p=0.3, max_alpha=0.6)
    #x = random_contrast_np(x, lower=0.8, upper=1.25)

    return x  # your existing preprocessing

def preprocess_input(x):
    return x  # your existing preprocessing
