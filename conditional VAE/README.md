# Conditional VAE (CVAE) on MNIST

A simple implementation of a **Conditional Variational Autoencoder (CVAE)** using TensorFlow/Keras and the MNIST handwritten digit dataset.

The main goal of this project is to understand how a VAE can be extended with a **condition** so that the generated output can be controlled.

For this project, the condition is the **MNIST digit label (0–9)**.

---

## 📌 Project Overview

A standard Variational Autoencoder learns:

```text
Image → Encoder → Latent Vector → Decoder → Image
```

A Conditional VAE additionally receives a condition:

```text
Image + Label
      ↓
   Encoder
      ↓
 Latent Vector
      ↓
Decoder + Label
      ↓
 Generated Image
```

For example:

```text
Random latent vector + label 7
              ↓
           Decoder
              ↓
       Generated digit 7
```

This allows us to control which digit the model generates.

---

## 🎯 Objectives

This project demonstrates:

- Loading the MNIST dataset
- Image normalization
- Image flattening
- One-hot encoding of labels
- Building an Encoder
- Building a Decoder
- Creating a custom Sampling layer
- Understanding the reparameterization trick
- Building a Conditional VAE
- Implementing Reconstruction Loss
- Implementing KL Divergence Loss
- Creating a custom training step
- Training the CVAE with Adam
- Generating a new digit conditionally

---

## 🧠 What is a Conditional VAE?

A Conditional Variational Autoencoder is a VAE that uses additional information called a **condition**.

In this implementation:

```text
Input image  → MNIST handwritten digit
Condition    → Digit label (0–9)
Latent size  → 16
```

The encoder receives:

```text
Image + Label
```

and learns a latent distribution:

```text
z_mean
z_log_var
```

A latent vector `z` is sampled from this distribution.

The decoder then receives:

```text
z + Label
```

and reconstructs/generates the image.

---

## 🏗️ Architecture

```text
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
             ┌───────────┴───────────┐
             │                       │
             │                    Label
             │                       │
             │                  One-hot (10)
             │                       │
             └───────────┬───────────┘
                         ▼
                    Concatenate
                         │
                         ▼
                     794 values
                         │
                         ▼
                    Dense(256)
                       ReLU
                         │
                 ┌───────┴────────┐
                 ▼                ▼
              z_mean          z_log_var
                 │                │
                 └───────┬────────┘
                         ▼
                      Sampling
                         │
                         ▼
                    z (16 values)
                         │
                         │
             ┌───────────┴───────────┐
             │                       │
             │                    Label
             │                       │
             └───────────┬───────────┘
                         ▼
                    Concatenate
                         │
                         ▼
                    Dense(256)
                       ReLU
                         │
                         ▼
                    Dense(784)
                      Sigmoid
                         │
                         ▼
                  Generated Image
                     28 × 28
```

---

## 🔑 Key Components

### 1. Encoder

The encoder receives:

```text
Image + Label
```

The flattened image contains:

```text
784 values
```

and the one-hot label contains:

```text
10 values
```

Therefore:

```text
784 + 10 = 794 inputs
```

The encoder produces:

```text
z_mean
z_log_var
```

and uses the Sampling layer to produce:

```text
z
```

---

### 2. Sampling Layer

The Sampling layer implements the VAE reparameterization trick:

\[
z = \mu + \sigma\epsilon
\]

where:

- `μ` = `z_mean`
- `σ` = `exp(0.5 × z_log_var)`
- `ε` = random value sampled from a standard normal distribution

In code:

```python
z_mean + tf.exp(0.5 * z_log_var) * epsilon
```

This allows stochastic sampling while keeping the operation differentiable for backpropagation.

---

### 3. Decoder

The decoder receives:

```text
Latent vector z + Label
```

The latent vector contains:

```text
16 values
```

The label contains:

```text
10 values
```

Therefore:

```text
16 + 10 = 26 inputs
```

The decoder produces:

```text
784 values
```

which are reshaped back into:

```text
28 × 28
```

---

## 📉 Loss Function

The CVAE uses two major loss components.

### Reconstruction Loss

The reconstruction loss measures how closely the generated image matches the original image.

```python
reconstruction_loss = tf.reduce_mean(
    tf.keras.losses.binary_crossentropy(
        image,
        reconstructed
    )
)
```

The goal is:

```text
Original image ≈ Reconstructed image
```

---

### KL Divergence Loss

The KL loss encourages the learned latent distribution to remain close to a standard normal distribution:

\[
N(0,I)
\]

The implementation is:

```python
kl_loss = -0.5 * tf.reduce_mean(
    1 + z_log_var
    - tf.square(z_mean)
    - tf.exp(z_log_var)
)
```

---

### Total Loss

The complete objective is:

\[
\boxed{
Loss = Reconstruction\ Loss + KL\ Loss
}
\]

The reconstruction term encourages accurate images, while the KL term regularizes the latent space.

---

## 🔄 Training Flow

```text
             Image + Label
                   │
                   ▼
                Encoder
                   │
            ┌──────┴──────┐
            ▼             ▼
         z_mean       z_log_var
            │             │
            └──────┬──────┘
                   ▼
               Sampling
                   │
                   ▼
                   z
                   │
                   ├──────────────┐
                   │              │
                   │            Label
                   │              │
                   └──────┬───────┘
                          ▼
                       Decoder
                          │
                          ▼
                  Reconstructed Image
                          │
                          ▼
                 Reconstruction Loss

             z_mean + z_log_var
                       │
                       ▼
                    KL Loss

                Total Loss
```

---

## 🚀 Generation Flow

After training, a random latent vector is sampled:

```python
z = np.random.normal(
    size=(1, latent_dim)
)
```

A desired digit is selected:

```python
label = tf.keras.utils.to_categorical(
    [7],
    num_classes=10
)
```

The decoder receives both:

```text
Random z + label 7
```

and generates a digit.

```python
generated = decoder.predict([z, label])
```

The generated vector is then reshaped:

```python
generated = generated.reshape(28, 28)
```

and displayed using Matplotlib.

---

## 📂 Project Structure

Recommended repository structure:

```text
Conditional-VAE/
│
├── Conditional_VAE.ipynb
├── Conditional_VAE_Code_Explanation.md
├── README.md
├── requirements.txt
└── images/
    └── generated_digit.png
```

### Files

| File | Purpose |
|---|---|
| `Conditional_VAE.ipynb` | Main executable notebook |
| `Conditional_VAE_Code_Explanation.md` | Detailed line-by-line explanation |
| `README.md` | Project documentation |
| `requirements.txt` | Python dependencies |
| `images/` | Optional generated-result images |

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Conditional-VAE
```

Create a virtual environment:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

Open:

```text
Conditional_VAE.ipynb
```

and execute the cells from top to bottom.

The MNIST dataset will be downloaded through Keras when the dataset-loading cell is executed.

---

## 📊 Training Configuration

The implementation uses:

```text
Dataset       : MNIST
Image size    : 28 × 28
Flattened     : 784
Classes       : 10
Latent dim    : 16
Hidden units  : 256
Epochs        : 10
Batch size    : 128
Optimizer     : Adam
Output        : 784 pixels
```

---

## 🧪 Example: Generate Digit 7

The condition can be changed to generate different digits.

For digit `7`:

```python
label = tf.keras.utils.to_categorical(
    [7],
    num_classes=10
)

generated = decoder.predict([z, label])
```

For digit `3`:

```python
label = tf.keras.utils.to_categorical(
    [3],
    num_classes=10
)
```

For digit `9`:

```python
label = tf.keras.utils.to_categorical(
    [9],
    num_classes=10
)
```

The key idea is:

```text
Same / different latent vectors
            +
      Different labels
            ↓
      Different classes
```

---

## 📚 Important Concepts Learned

This project is useful for understanding several important Generative AI concepts:

- Variational Autoencoders
- Conditional Generative Models
- Latent Spaces
- Encoder–Decoder Architecture
- Probability Distributions
- Reparameterization Trick
- KL Divergence
- Reconstruction Loss
- Sampling
- Neural Network Optimization
- Custom Keras Layers
- Custom Training Steps

---

## 🧩 VAE vs CVAE

### VAE

```text
Image
  ↓
Encoder
  ↓
Latent z
  ↓
Decoder
  ↓
Image
```

### CVAE

```text
Image + Condition
       ↓
    Encoder
       ↓
     Latent z
       ↓
Decoder + Condition
       ↓
     Image
```

In this project:

```text
Condition = MNIST digit label
```

Therefore:

```text
z + label 0 → digit 0
z + label 5 → digit 5
z + label 7 → digit 7
```

---

## 📖 Detailed Explanation

For a detailed, line-by-line explanation of the implementation, see:

**`Conditional_VAE_Code_Explanation.md`**

That document explains:

- Python keywords
- TensorFlow/Keras functions
- Every major code block
- Why each component is used
- Encoder
- Decoder
- Sampling
- Reparameterization
- Loss functions
- Training
- Conditional generation

---

## 📝 Learning Note

This repository is primarily intended as a **learning and educational implementation** of a Conditional VAE.

The architecture is deliberately simple so that the underlying CVAE concepts can be understood clearly before moving to more complex architectures such as:

- Convolutional VAEs
- Conditional image generation
- Diffusion models
- Stable Diffusion
- Latent Diffusion Models
- More advanced generative architectures

---

## ⭐ Key Takeaway

The central idea of this project is:

> **A Conditional VAE learns a latent distribution from an image while considering its label, then uses a sampled latent vector plus a requested label to generate a new image belonging to that class.**

Mathematically:

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

and:

\[
\boxed{
Loss =
Reconstruction\ Loss + KL\ Divergence
}
\]

---

## 👨‍💻 Author

**Abhi / Tutor Abhi**

Educational implementation and learning project focused on Machine Learning, Deep Learning and Generative AI.
