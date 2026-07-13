"""
=============================================================================================================
Python script to demonstrate a single-image Variational Autoencoder (VAE)
=============================================================================================================
This program demonstrates image compression using a variational autoencoder. It's intentionally simplified
for clarity rather than performance.

SCRIPT OVERVIEW
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Running this script will
1. Load the image from 'files/original.jpg'
2. Resize and convert it to a tensor.
3. Build a small VAE (encoder and decoder) from scratch in PyTorch.
4. Intentionally overfit the VAE to this (original.jpg) image in the
   training section below
5. Use the trained VAE to "compress" (encode) and "decompress" (decode/reconstruct) the image.
6. Save the reconstruction to 'files/reconstructed.png'
7. Display the reconstructed and original image side-by-side.
8. Print conceptual compression statistics.


Requirements:
    !pip install matplotlib numpy pillow torch torchvision tqdm

Author: Nyanjui
Date: dd mmmm yyyy
"""
# --------------------------------------------------------------------------------
# 0. Import required modules
# --------------------------------------------------------------------------------
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as functional

from pathlib import Path
from PIL import Image
from torch import Tensor
from torch.optim import Adam
from torchvision import  transforms
from tqdm import tqdm
from typing import Tuple



import warnings

# Suppress warnings for cleaner output demo
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------
# 1. Constants
# --------------------------------------------------------------------------------
# Directory containing the input image and where the outputs will be written
FILES_DIRECTORY: Path = Path("../files")

# Path to the input image
ORIGINAL_IMAGE_PATH: Path = FILES_DIRECTORY / "original.jpg"

# Path where the reconstructed image will be saved
RECONSTRUCTED_IMAGE_PATH: Path = FILES_DIRECTORY / "reconstructed.png"

# Resize the image to a fixed square resolution. A fixed size keeps the
# network architecture simple, since fully connected layers require a known, constant input value
IMAGE_SIZE: int = 128

# Number of colour channels. We work with RGB, hence 3 channels
NUMBER_OF_CHANNELS: int = 3

# Dimensionality of the VAE's latent space (the "compressed" representation of the image) 64 is a good value
LATENT_DIMENSION: int = 64

# Sizes of the hidden fully connected layers in the encoder and decoder
HIDDEN_LAYER_ONE_SIZE: int = 1024
HIDDEN_LAYER_TWO_SIZE: int = 256

# Training hyperparameters
NUMBER_OF_EPOCHS: int = 200
LEARNING_RATE: float = 1e-3

# Weight applied to the KL_DIVERGENCE term in the loss function.
KL_DIVERGENCE_WEIGHT: float = 1.0 # Set to 1.0 for a "vanilla" VAE

# Processing device on which all tensors and models will live. Prefer GPU (CUDA) if available or CPU if not
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fixed random seed for reproducibility from run to run
RANDOM_SEED: int = 42

# --------------------------------------------------------------------------------
# 2. Utility Function
# --------------------------------------------------------------------------------
def set_random_seed(seed: int) -> None:
    """
    Set the random seed for Numpy and PyTorch.

    :param seed:
        The integer seed to apply to Numpy's and PyTorch's random number generators.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_and_preprocess_image(image_path: Path, image_size:int) -> Tensor:
    """Load an image from disk and convert it into a normalised tensor.

        This function demonstrates the standard "load, resize, convert to
        tensor" pipeline that precedes almost all computer vision deep
        learning models.

        Parameters
        ----------
        image_path:
            The path to the JPEG (or other Pillow-readable) image file.
        image_size:
            The width and height, in pixels, to resize the (square) image to.

        Returns
        -------
        Tensor
            A tensor of shape ``(1, channels, image_size, image_size)`` with
            pixel values scaled to the range ``[0, 1]``. The leading dimension
            of size 1 is the "batch" dimension that PyTorch models expect,
            even though we only have a single image.

        Raises
        ------
        FileNotFoundError
            If no image exists at ``image_path``.
        """
    if not image_path.exists():
        raise FileNotFoundError(f"Could not find input image at '{image_path}'"
                                f"Please place a JPEG named 'original.jpg' at {ORIGINAL_IMAGE_PATH}"
                                f" before running this script/program.")

    # Pillow loads the raw image file; we force conversion to RGB in case the source image
    # is greyscale or has an alpha channel
    pil_image = Image.open(image_path).convert('RGB')

    # torchvision.transforms.Compose lets us chain several preprocessing
    # steps together. Resize changes the spatial resolution; ToTensor
    # converts the Pillow image into a PyTorch tensor and automatically
    # rescales pixel values from [0, 255] integers to [0.0, 1.0] floats,
    # which is the range our network's sigmoid output will also use.
    preprocessing_pipeline = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    image_tensor = preprocessing_pipeline(pil_image)

    # Add a batch dimension: shape becomes (1, C, H, W). PyTorch layers
    # such as nn.Linear and nn.Conv2d always expect a batch dimension,
    # even when the batch contains only one example.
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor.to(DEVICE)

def tensor_to_numpy_image(image_tensor: Tensor) -> np.ndarray:
    """Convert a normalised image tensor back into a displayable array.

        Parameters
        ----------
        image_tensor:
            A tensor of shape ``(1, channels, height, width)`` with values in
            the range ``[0, 1]``.

        Returns
        -------
        numpy.ndarray
            An array of shape ``(height, width, channels)`` with values in
            ``[0, 1]``, suitable for display with ``matplotlib``.
        """
    # Detach from the autograd graph, move to the CPU, and drop the batch
    # dimension before rearranging the axes for matplotlib, which expects
    # channels last (H, W, C) rather than PyTorch's channels-first (C, H, W).
    image = image_tensor.detach().cpu().squeeze(0)
    image = image.permute(1,2,0).numpy()
    image = np.clip(image, 0.0, 1.0)
    return image

def format_output(content:str)-> None:
    """Print a neatly formatted heading describing the current stage.

        Used throughout the program to give students a clear, sequential
        narrative of what the code is doing at any given moment - this is
        intended for projection during a lecture or lab session.

        Parameters
        ----------
        content:
            A short human-readable description of the stage about to begin.
        """
    print(f"\n" + "=" * 70)
    print(content)
    print(f"=" * 70)

# --------------------------------------------------------------------------------
# 3. Encoder class
# --------------------------------------------------------------------------------
class Encoder(nn.Module):
    """The encoder half of the Variational Autoencoder.

        The encoder's job is to compress an input image into a low-dimensional
        latent representation. Unlike a standard (deterministic) autoencoder,
        a VAE's encoder does not output a single latent vector directly.
        Instead, it outputs the *parameters of a probability distribution*
        (specifically, a diagonal Gaussian) over the latent space: a mean
        vector ``mu`` and a log-variance vector ``log_var``.

        Why output a distribution rather than a single point?
        -------------------------------------------------------
        If we only learned a single latent vector per image (as a plain
        autoencoder does), the latent space could become an arbitrary,
        disconnected scattering of points with "gaps" that do not correspond
        to any realistic image. By instead forcing each image to map to a
        small *region* (a Gaussian "cloud") of latent space, and by
        encouraging nearby regions to overlap (via the KL divergence term
        explained later), the latent space becomes smooth and continuous.
        This means that points we have never explicitly trained on - for
        example, points interpolated between two training images - still
        decode into plausible-looking images.

        We predict ``log_var`` rather than ``var`` directly for numerical
        convenience: the network's raw output can be any real number,
        whereas a variance must be positive. Exponentiating ``log_var``
        guarantees a positive variance without needing extra constraints
        on the network's weights.
        """
    def __init__(self,image_size: int, number_of_channels: int,
                 latent_dimension:int) -> None:
        """
            Construct the encoder network.

                Parameters
                ----------
                image_size:
                    The height and width (in pixels) of the square input images.
                number_of_channels:
                    The number of colour channels in the input images (3 for RGB).
                latent_dimension:
                    The size of the latent vector the encoder should produce.
        """
        super().__init__()

        # We flatten the image into a single long vector and use fully connected
        # ("dense") layers throughout. Convolutional layers are usually preferred
        # for images, but a fully connected network is easier since every connection
        # and shape is explicit.
        self.input_dimension = number_of_channels * image_size * image_size

        # A stack of two hidden layers gradually reduces the dimensionality
        # from the raw pixel count down towards the latent dimension.
        # ReLU is used as our non-linearity - a standard, simple choice.
        self.shared_hidden_layers = nn.Sequential(
            nn.Linear(self.input_dimension, HIDDEN_LAYER_ONE_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_ONE_SIZE, HIDDEN_LAYER_TWO_SIZE),
            nn.ReLU(),
        )

        # Two separate output "heads" branch off from the shared hidden
        # representation: one predicts the mean (mu) of the latent
        # distribution, and the other predicts the log-variance
        # (log_var). Both have the same shape: (latent_dimension,).
        self.mean_head = nn.Linear(HIDDEN_LAYER_TWO_SIZE, latent_dimension)
        self.log_variance_head = nn.Linear(HIDDEN_LAYER_TWO_SIZE, latent_dimension)

    def forward(self, image_batch: Tensor) -> Tuple[Tensor, Tensor]:
        """Run a forward pass through the encoder.

        Parameters
        ----------
        image_batch:
            A batch of images with shape
            ``(batch_size, channels, height, width)``.

        Returns
        -------
        Tuple[Tensor, Tensor]
            A tuple ``(mu, log_var)``, each of shape
            ``(batch_size, latent_dimension)``, describing the mean and
            log-variance of the approximate posterior distribution
            ``q(z | x)`` for every image in the batch.
        """
        batch_size = image_batch.shape[0]

        # Flatten each image in the batch from (C, H, W) into a single
        # long vector, ready for the fully connected layers.
        flattened_images = image_batch.view(batch_size, -1)

        hidden_representation = self.shared_hidden_layers(flattened_images)

        mean = self.mean_head(hidden_representation)
        log_variance = self.log_variance_head(hidden_representation)

        return mean, log_variance

# --------------------------------------------------------------------------------
# 4. Decoder class
# --------------------------------------------------------------------------------
class Decoder(nn.Module):
    """The decoder half of the Variational Autoencoder.

    The decoder takes a latent vector ``z`` (sampled from the
    distribution produced by the encoder) and reconstructs an image from
    it. Architecturally, it mirrors the encoder: the same layer sizes are
    used, but in reverse, gradually expanding the compact latent vector
    back up to the full number of pixels.
    """

    def __init__(
        self,
        image_size: int,
        number_of_channels: int,
        latent_dimension: int,
    ) -> None:
        """Construct the decoder network.

        Parameters
        ----------
        image_size:
            The height and width (in pixels) of the square output images.
        number_of_channels:
            The number of colour channels to reconstruct (3 for RGB).
        latent_dimension:
            The size of the latent vector this decoder accepts as input.
        """
        super().__init__()

        self.image_size = image_size
        self.number_of_channels = number_of_channels
        self.output_dimension = number_of_channels * image_size * image_size

        # The decoder is the mirror image of the encoder: it expands the latent
        # vector back up through the same hidden layer sizes, in reverse order,
        # before producing a full-size image vector.
        self.network = nn.Sequential(
            nn.Linear(self.output_dimension, HIDDEN_LAYER_TWO_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_TWO_SIZE, HIDDEN_LAYER_ONE_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_ONE_SIZE, self.output_dimension),
            nn.Sigmoid(),
        )

    def forward(self, latent_vector_batch: Tensor) -> Tensor:
        """
            Run a forward pass through the decoder.

            Parameters
            ----------
            latent_vector_batch:
                A batch of latent vectors with shape
                ``(batch_size, latent_dimension)``.

            Returns
            -------
            Tensor
                A batch of reconstructed images with shape
                ``(batch_size, channels, height, width)`` and pixel values in
                the range ``[0, 1]``.
        """
        batch_size = latent_vector_batch.shape[0]
        flat_reconstruction = self.network(latent_vector_batch)

        # Reshape the flat vector of pixel values back into a proper image
        # tensor: (batch_size, channels, height, width)
        reconstructed_images = flat_reconstruction.view(batch_size,
                                                        self.number_of_channels, self.image_size, self.image_size)
        return reconstructed_images

# --------------------------------------------------------------------------------
# 5. Variational Autoencoder class
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# xx. Main Execution Function
# --------------------------------------------------------------------------------
def main() -> None:
    pass


# --------------------------------------------------------------------------------
# yy. Run the script by invoking it's main() function
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    main()