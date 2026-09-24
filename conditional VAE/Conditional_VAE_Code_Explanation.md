# Conditional VAE (CVAE) — Code Explanation

This document explains the Conditional Variational Autoencoder (CVAE) implementation used with the MNIST dataset. It is intended to be saved alongside the main notebook/code in GitHub as a learning/reference document.

---

## 1. What is a Conditional VAE?

A Conditional VAE is a Variational Autoencoder that receives an additional piece of information called a **condition**.

In this implementation:

- Input image = handwritten MNIST digit
- Condition = digit label (`0` to `9`)
- Latent dimension = `16`

Normal VAE:

```text
image → encoder → z → decoder → image
```

Conditional VAE:

```text
image + label → encoder → z
z + label → decoder → reconstructed image
```

The condition allows us to control what class the decoder generates.

For example:

```text
random latent vector + label 7 → generated digit 7
```

---

# 2. Overall Architecture

```text
                   TRAINING

             MNIST IMAGE
             28 × 28
                │
                ▼
             Flatten
                │
                ▼
            784 values
                │
                │
                ├──────────────┐
                │              │
                │          Digit label
                │              │
                │              ▼
                │       One-hot 10 values
                │              │
                └──────┬───────┘
                       ▼
                  CONCATENATE
                       │
                       ▼
                  794 values
                       │
                       ▼
                  Dense 256
                       │
                       ▼
                 ┌─────┴─────┐
                 ▼           ▼
              z_mean      z_log_var
                 │           │
                 └─────┬─────┘
                       ▼
                   Sampling
                       │
                       ▼
                  z (16 values)
                       │
                       │
             ┌─────────┴─────────┐
             │                   │
             │                 label
             │                   │
             ▼                   ▼
                 CONCATENATE
                       │
                       ▼
                  Dense 256
                       │
                       ▼
                  Dense 784
                    sigmoid
                       │
                       ▼
                Reconstructed
                    image
```

---

# 3. Importing Libraries

```python
import tensorflow as tf
from keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt
```

## `import`

`import` is a Python keyword used to bring functionality from another library into the program.

## `import tensorflow as tf`

TensorFlow is the machine-learning framework used here.

`tf` is an alias, so instead of writing:

```python
tensorflow.keras
```

we can write:

```python
tf.keras
```

## `from keras import layers, Model`

This imports:

- `layers` — used to create neural-network layers such as `Dense`, `Input`, and `Concatenate`
- `Model` — used to create Keras models

## `import numpy as np`

NumPy is used for numerical operations.

`np` is the conventional alias.

Example:

```python
np.random.normal()
```

## `import matplotlib.pyplot as plt`

Matplotlib is used to display the generated digit.

Example:

```python
plt.imshow(generated, cmap="gray")
plt.show()
```

---

# 4. Loading MNIST

```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
```

MNIST contains handwritten digits from `0` to `9`.

Each image is:

```text
28 × 28 = 784 pixels
```

The variables are:

- `x_train` — training images
- `y_train` — training labels
- `x_test` — test images
- `y_test` — test labels

For example:

```text
x_train[0] → image
y_train[0] → corresponding digit label
```

If `x_train[0]` is an image of `5`, then:

```text
x_train[0] → digit 5
y_train[0] → 5
```

---

# 5. Normalizing the Images

```python
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
```

## `.astype("float32")`

Converts the pixel values into 32-bit floating-point numbers.

## `/255.0`

MNIST pixels originally range from:

```text
0 → 255
```

After normalization:

```text
0 → 1
```

Examples:

```text
0   / 255 = 0.0
128 / 255 ≈ 0.502
255 / 255 = 1.0
```

This is useful because the decoder uses a sigmoid output, which also produces values between `0` and `1`.

---

# 6. Flattening the Images

```python
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)
```

Originally:

```text
28 × 28
```

After reshaping:

```text
784
```

because:

```text
28 × 28 = 784
```

The `-1` tells NumPy/TensorFlow:

> Automatically determine the number of images.

For example:

```text
(60000, 28, 28)
```

becomes:

```text
(60000, 784)
```

The actual pixel data is not changed; only its shape is changed.

---

# 7. One-Hot Encoding the Labels

```python
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
```

MNIST has 10 classes:

```text
0 1 2 3 4 5 6 7 8 9
```

A label such as:

```text
7
```

is converted into:

```text
[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
```

Similarly:

```text
0 → [1,0,0,0,0,0,0,0,0,0]
1 → [0,1,0,0,0,0,0,0,0,0]
2 → [0,0,1,0,0,0,0,0,0,0]
...
9 → [0,0,0,0,0,0,0,0,0,1]
```

`num_classes=10` specifies that there are ten possible classes.

---

# 8. Latent Dimension and Number of Classes

```python
latent_dim = 16
num_classes = 10
```

## `latent_dim = 16`

The encoder compresses the image information into a 16-dimensional latent vector.

Instead of:

```text
784 values
```

the latent representation contains:

```text
16 values
```

## `num_classes = 10`

There are ten digit classes:

```text
0–9
```

Therefore the one-hot condition contains 10 values.

---

# 9. Sampling Layer

```python
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs

        epsilon = tf.random.normal(
            shape=tf.shape(z_mean)
        )

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
```

This is one of the most important parts of the VAE.

## `class Sampling(layers.Layer):`

Creates a custom Keras layer named `Sampling`.

It inherits from `layers.Layer`, so it behaves like a normal neural-network layer.

## `def call(self, inputs):`

`call()` defines what happens when data passes through the layer.

## `z_mean, z_log_var = inputs`

The sampling layer receives:

- `z_mean` — mean of the latent distribution
- `z_log_var` — logarithm of the latent variance

The encoder predicts both.

## `epsilon`

```python
epsilon = tf.random.normal(
    shape=tf.shape(z_mean)
)
```

Generates random values from a standard normal distribution:

```text
N(0, 1)
```

For example:

```text
[0.21, -1.13, 0.47, ...]
```

## Reparameterization trick

The returned value is:

```python
z_mean + tf.exp(0.5 * z_log_var) * epsilon
```

Mathematically:

\[
z = \mu + \sigma\epsilon
\]

where:

```text
μ = z_mean
σ = exp(0.5 × z_log_var)
ε = random normal noise
```

This allows the model to sample from a distribution while still allowing gradients to flow through the operation during training.

---

# 10. Encoder

```python
image_input = layers.Input(shape=(784,))
label_input = layers.Input(shape=(num_classes,))
```

The encoder has two inputs.

## Image input

```python
image_input = layers.Input(shape=(784,))
```

The image has 784 flattened pixel values.

## Label input

```python
label_input = layers.Input(shape=(num_classes,))
```

The condition has 10 values because `num_classes = 10`.

---

# 11. Combining Image and Label

```python
encoder_input = layers.Concatenate()(
    [image_input, label_input]
)
```

The image contains:

```text
784 values
```

The label contains:

```text
10 values
```

After concatenation:

```text
784 + 10 = 794 values
```

Conceptually:

```text
image [784]
     +
label [10]
     ↓
combined input [794]
```

This is how the encoder receives the condition.

---

# 12. Encoder Dense Layer

```python
x = layers.Dense(256, activation="relu")(encoder_input)
```

## `Dense(256)`

Creates a fully connected layer containing 256 neurons.

The data goes from:

```text
794 values
   ↓
256 neurons
```

## `activation="relu"`

ReLU is:

\[
ReLU(x) = max(0,x)
\]

Examples:

```text
-5 → 0
-2 → 0
0  → 0
3  → 3
7  → 7
```

The activation introduces non-linearity so the neural network can learn complex relationships.

---

# 13. Latent Mean

```python
z_mean = layers.Dense(latent_dim)(x)
```

Because:

```python
latent_dim = 16
```

this produces 16 values.

These values represent:

\[
\mu
\]

the mean of the latent distribution.

---

# 14. Latent Log Variance

```python
z_log_var = layers.Dense(latent_dim)(x)
```

Again, 16 values are produced.

They represent:

\[
\log(\sigma^2)
\]

instead of directly predicting variance.

The logarithm is convenient because variance must be positive, while its logarithm can take any real value.

Later:

```python
tf.exp(z_log_var)
```

converts it back to a variance-related quantity.

---

# 15. Sampling the Latent Vector

```python
z = Sampling()([z_mean, z_log_var])
```

The encoder produces:

```text
z_mean
z_log_var
```

The Sampling layer uses both to create:

```text
z
```

The latent vector has 16 dimensions.

The encoder therefore performs:

```text
image + label
      ↓
encoder
      ↓
z_mean + z_log_var
      ↓
sampling
      ↓
z
```

---

# 16. Creating the Encoder Model

```python
encoder = Model(
    [image_input, label_input],
    [z_mean, z_log_var, z],
    name="Encoder"
)
```

The encoder has two inputs:

```text
image
label
```

and three outputs:

```text
z_mean
z_log_var
z
```

So conceptually:

```text
Encoder(image, label)
        ↓
(mean, log_variance, latent_vector)
```

---

# 17. Decoder Inputs

```python
latent_input = layers.Input(shape=(latent_dim,))
label_decoder = layers.Input(shape=(num_classes,))
```

The decoder also has two inputs:

```text
latent vector
condition/label
```

The latent vector contains:

```text
16 values
```

The label contains:

```text
10 values
```

---

# 18. Combining Latent Vector and Label

```python
decoder_input = layers.Concatenate()(
    [latent_input, label_decoder]
)
```

The decoder combines:

```text
16 latent values
+
10 label values
=
26 values
```

This is important because the decoder needs both:

1. The latent information
2. The desired digit class

---

# 19. Decoder Hidden Layer

```python
x = layers.Dense(256, activation="relu")(decoder_input)
```

The 26 values are passed through a Dense layer with 256 neurons.

```text
26
 ↓
256
 ↓
ReLU
```

---

# 20. Decoder Output

```python
decoder_output = layers.Dense(
    784, activation="sigmoid"
)(x)
```

The decoder needs to reconstruct the original image.

Since the flattened MNIST image has 784 pixels:

```text
28 × 28 = 784
```

the output layer contains 784 neurons.

## Why sigmoid?

Sigmoid produces values between 0 and 1.

That matches the normalized image values:

```text
0 → 1
```

---

# 21. Creating the Decoder Model

```python
decoder = Model(
    [latent_input, label_decoder],
    decoder_output,
    name="Decoder"
)
```

The decoder performs:

```text
z + label
    ↓
Decoder
    ↓
784 pixel values
```

---

# 22. CVAE Class

```python
class CVAE(Model):
```

This creates a custom Keras model called `CVAE`.

It inherits from `Model`.

The CVAE contains:

```text
Encoder
Decoder
```

---

# 23. Constructor

```python
def __init__(self, encoder, decoder):
    super().__init__()

    self.encoder = encoder
    self.decoder = decoder
```

## `__init__`

Python's constructor. It runs when the CVAE object is created.

## `super().__init__()`

Initializes the parent Keras `Model`.

## `self.encoder = encoder`

Stores the encoder inside the CVAE object.

## `self.decoder = decoder`

Stores the decoder inside the CVAE object.

---

# 24. CVAE Forward Pass

```python
def call(self, inputs):
    image, label = inputs

    z_mean, z_log_var, z = self.encoder([image, label])

    reconstructed = self.decoder([z, label])

    return reconstructed
```

The forward pass is:

```text
image + label
      ↓
Encoder
      ↓
z_mean, z_log_var, z
      ↓
Decoder(z + label)
      ↓
reconstructed image
```

The condition is used by both encoder and decoder.

---

# 25. Custom Training Step

```python
def train_step(self, data):
    (image, label) = data[0]
```

A VAE has a special loss consisting of:

```text
Total Loss
=
Reconstruction Loss
+
KL Divergence Loss
```

Therefore the model defines its own `train_step()`.

---

# 26. GradientTape

```python
with tf.GradientTape() as tape:
```

`GradientTape` records the mathematical operations during the forward pass.

After calculating the loss, TensorFlow can use the recorded operations to calculate gradients.

Conceptually:

```text
Forward pass
     ↓
Calculate loss
     ↓
Calculate gradients
     ↓
Update weights
```

---

# 27. Encode During Training

```python
z_mean, z_log_var, z = self.encoder([image, label])
```

The image and condition are passed to the encoder.

Output:

```text
z_mean
z_log_var
z
```

---

# 28. Reconstruct the Image

```python
reconstructed = self.decoder([z, label])
```

The sampled latent vector and the original label are passed to the decoder.

Output:

```text
reconstructed image
```

---

# 29. Reconstruction Loss

```python
reconstruction_loss = tf.reduce_mean(
    tf.keras.losses.binary_crossentropy(
        image,
        reconstructed
    )
)
```

The reconstruction loss measures how well the decoder reconstructs the original image.

Conceptually:

```text
Original image
      ↓
compare
      ↑
Reconstructed image
```

The model tries to make:

```text
reconstructed ≈ original
```

## Binary Cross Entropy

Binary cross entropy compares the target pixel values with predicted pixel values.

This is appropriate here because:

- Images are normalized to `0–1`
- Decoder output uses sigmoid and therefore produces `0–1`

## `reduce_mean`

There are many loss values. `reduce_mean` averages them into a single loss value.

---

# 30. KL Divergence Loss

```python
kl_loss = -0.5 * tf.reduce_mean(
    1 + z_log_var
    - tf.square(z_mean)
    - tf.exp(z_log_var)
)
```

The KL divergence term encourages the learned latent distribution to stay close to a standard normal distribution:

\[
N(0,I)
\]

The formula is:

\[
-\frac{1}{2}
\left(
1+\log(\sigma^2)-\mu^2-\sigma^2
\right)
\]

The code corresponds to:

```text
μ                 → z_mean
log(σ²)            → z_log_var
μ²                 → tf.square(z_mean)
σ²                 → tf.exp(z_log_var)
```

Why is this useful?

Without KL regularization, the latent space could become poorly organized.

KL divergence encourages a smooth latent space so that random latent vectors can be sampled for generation.

---

# 31. Total Loss

```python
total_loss = reconstruction_loss + kl_loss
```

The CVAE optimizes:

\[
Loss =
Reconstruction\ Loss + KL\ Loss
\]

The two components have different purposes:

### Reconstruction loss

Make the output look like the original image.

### KL loss

Keep the latent distribution well behaved.

---

# 32. Calculating Gradients

```python
grads = tape.gradient(
    total_loss,
    self.trainable_weights
)
```

This asks TensorFlow:

> How should each trainable parameter change to reduce the total loss?

The result is a list of gradients corresponding to the model's trainable weights.

---

# 33. Updating Weights

```python
self.optimizer.apply_gradients(
    zip(grads, self.trainable_weights)
)
```

`zip()` pairs each gradient with its corresponding weight.

Conceptually:

```text
gradient 1 ↔ weight 1
gradient 2 ↔ weight 2
gradient 3 ↔ weight 3
...
```

The optimizer uses these pairs to update the model parameters.

---

# 34. Returning Training Metrics

```python
return {
    "loss": total_loss,
    "reconstruction_loss": reconstruction_loss,
    "kl_loss": kl_loss,
}
```

This tells Keras to report:

- Total loss
- Reconstruction loss
- KL loss

during training.

---

# 35. Creating the CVAE

```python
cvae = CVAE(encoder, decoder)
```

Creates the complete model:

```text
CVAE
 ├── Encoder
 └── Decoder
```

---

# 36. Compiling the Model

```python
cvae.compile(optimizer="adam")
```

The model uses the Adam optimizer.

No explicit loss function is passed to `compile()` because the loss is calculated manually inside the custom `train_step()`.

---

# 37. Training

```python
history = cvae.fit(
    [x_train, y_train],
    epochs=10,
    batch_size=128
)
```

## `fit()`

Starts model training.

## `[x_train, y_train]`

The CVAE has two inputs:

```text
images
labels
```

so both are supplied.

## `epochs=10`

The entire training dataset is processed ten times.

## `batch_size=128`

The model processes 128 examples at a time before updating the weights.

For approximately 60,000 MNIST training images:

```text
60000 / 128 ≈ 469 batches
```

which explains the training progress showing approximately 469 steps per epoch.

---

# 38. Generating a New Digit

After training, a random latent vector is created:

```python
z = np.random.normal(
    size=(1, latent_dim)
)
```

Because:

```python
latent_dim = 16
```

the generated latent vector has shape:

```text
(1, 16)
```

The values are sampled from a standard normal distribution.

---

# 39. Choosing the Digit to Generate

```python
label = tf.keras.utils.to_categorical(
    [7],
    num_classes=10
)
```

This creates the one-hot representation of digit 7:

```text
[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
```

So we are telling the decoder:

> Generate a digit belonging to class 7.

---

# 40. Generate the Image

```python
generated = decoder.predict(
    [z, label]
)
```

The decoder receives:

```text
random latent vector
+
label 7
```

and generates:

```text
784 pixel values
```

Conceptually:

```text
Random z
   +
label = 7
   ↓
Decoder
   ↓
784 pixels
```

---

# 41. Convert Back to 28×28

```python
generated = generated.reshape(28, 28)
```

The decoder produced:

```text
784 values
```

but an image needs:

```text
28 × 28
```

So:

```text
784
 ↓
28 × 28
```

---

# 42. Display the Generated Digit

```python
plt.imshow(generated, cmap="gray")
plt.show()
```

`imshow()` displays the image.

`cmap="gray"` displays it in grayscale.

`plt.show()` renders the plot.

The final result should be a generated handwritten digit corresponding to the requested condition.

---

# 43. Complete CVAE Flow

The complete training flow is:

```text
                    IMAGE + LABEL
                         │
                         ▼
                    ENCODER
                         │
                  ┌──────┴──────┐
                  ▼             ▼
               z_mean       z_log_var
                  │             │
                  └──────┬──────┘
                         ▼
                     SAMPLING
                         │
                         ▼
                         z
                         │
                         ├──────────────┐
                         │              │
                         │            LABEL
                         │              │
                         └──────┬───────┘
                                ▼
                            DECODER
                                │
                                ▼
                       RECONSTRUCTED IMAGE
                                │
                                ▼
                       RECONSTRUCTION LOSS

z_mean + z_log_var
        │
        ▼
    KL LOSS

TOTAL LOSS
     =
RECONSTRUCTION LOSS
     +
   KL LOSS
```

---

# 44. VAE vs CVAE

## VAE

```text
image
  ↓
encoder
  ↓
z
  ↓
decoder
  ↓
image
```

## CVAE

```text
image + condition
       ↓
    encoder
       ↓
       z
       ↓
decoder + condition
       ↓
     image
```

In this implementation:

```text
condition = MNIST digit label
```

Therefore:

```text
random z + label 0 → generate 0
random z + label 5 → generate 5
random z + label 7 → generate 7
```

---

# 45. Why is the label provided to BOTH Encoder and Decoder?

The encoder receives:

```text
image + label
```

so it can learn a latent representation while knowing the class.

The decoder receives:

```text
z + label
```

so it knows which class should be generated.

Therefore:

```text
Encoder:
image + class → latent distribution

Decoder:
latent + class → generated image
```

This is the key idea behind the conditional architecture.

---

# 46. Important Terms Cheat Sheet

| Term | Meaning | Purpose |
|---|---|---|
| `import` | Load library | Use external functionality |
| `tf` | TensorFlow alias | Shorter TensorFlow name |
| `layers` | Keras layer collection | Build neural networks |
| `Model` | Keras model class | Create models |
| `np` | NumPy alias | Numerical operations |
| `latent_dim` | Latent vector size | 16-dimensional latent space |
| `num_classes` | Number of classes | 10 MNIST classes |
| `Input` | Defines input shape | Specify model input |
| `Dense` | Fully connected layer | Learn transformations |
| `relu` | Activation | Introduce non-linearity |
| `sigmoid` | Activation | Output values from 0 to 1 |
| `Concatenate` | Joins tensors | Combine data and condition |
| `Sampling` | Custom layer | Sample latent vector |
| `z_mean` | Latent mean | Mean of latent distribution |
| `z_log_var` | Log variance | Controls latent spread |
| `epsilon` | Random noise | Reparameterization |
| `Encoder` | Compression network | Input → latent distribution |
| `Decoder` | Generation network | Latent + label → image |
| `CVAE` | Conditional VAE | Complete architecture |
| `call()` | Forward pass | Defines computation |
| `train_step()` | Custom training logic | Implements VAE loss |
| `GradientTape` | Gradient recorder | Calculate gradients |
| `reconstruction_loss` | Reconstruction error | Make output resemble input |
| `kl_loss` | KL divergence | Regularize latent space |
| `optimizer` | Weight update algorithm | Train model |
| `Adam` | Optimizer | Update parameters |
| `fit()` | Training function | Train the model |
| `epoch` | One complete dataset pass | Repeat training |
| `batch_size` | Samples per update | Efficient training |
| `predict()` | Inference/generation | Produce output |
| `reshape()` | Change shape | 784 ↔ 28×28 |
| `imshow()` | Display image | Visualize output |

---

# 47. The Most Important Formula

The complete CVAE objective is:

\[
\boxed{
Loss =
Reconstruction\ Loss +
KL\ Divergence
}
\]

The architecture can be summarized as:

\[
\boxed{
(x,y)
\rightarrow Encoder
\rightarrow
(\mu,\log\sigma^2)
\rightarrow z
\rightarrow
Decoder(z,y)
\rightarrow \hat{x}
}
\]

where:

- \(x\) = input image
- \(y\) = condition/label
- \(\mu\) = latent mean
- \(\log\sigma^2\) = latent log variance
- \(z\) = sampled latent vector
- \(\hat{x}\) = reconstructed/generated image

---

# 48. One-Sentence Memory Trick

> **A Conditional VAE learns a latent distribution from an image while considering its label, then uses a sampled latent vector plus a requested label to generate a new image belonging to that class.**

For this MNIST implementation:

```text
Image + Digit Label
        ↓
      Encoder
        ↓
  Latent Distribution
        ↓
      Sampling
        ↓
    Latent Vector
        +
    Digit Label
        ↓
      Decoder
        ↓
 Generated Digit
```
