UCS654 Predictive Analysis Using Statistics – Assignment 2
Learning Probability Density Functions Using Roll-Number-Parameterized Non-Linear Model
1. Introduction

This assignment focuses on learning an unknown probability density function (PDF) using real-world air quality data. A roll-number-based nonlinear transformation is applied to the data, and a Generative Adversarial Network (GAN) is trained to learn the probability distribution of the transformed variable using samples only.

No parametric form (Gaussian, Exponential, etc.) of the probability density function is assumed.

2. Dataset Information

The dataset used is the India Air Quality Dataset obtained from Kaggle.

From the complete dataset, only the Nitrogen Dioxide (NO₂) concentration values are used, represented by the variable x (column: no2).

Missing values in the NO₂ column are removed before further processing.

3. Methodology

The approach consists of three main steps:

Step 1: Roll-Number-Based Nonlinear Transformation

Each NO₂ value x is transformed into a new variable z using:

𝑧
=
𝑥
+
𝑎
𝑟
⋅
sin
⁡
(
𝑏
𝑟
⋅
𝑥
)
z=x+a
r
	​

⋅sin(b
r
	​

⋅x)

Where the parameters are computed using the university roll number:

𝑎
𝑟
=
0.5
×
(
𝑟
 
m
o
d
 
7
)
a
r
	​

=0.5×(rmod7)
𝑏
𝑟
=
0.3
×
(
(
𝑟
 
m
o
d
 
5
)
+
1
)
b
r
	​

=0.3×((rmod5)+1)

For roll number 102317189, the parameters are:

Parameter	Value
r mod 7	2
r mod 5	4
a_r	1.0
b_r	1.5

This ensures a unique nonlinear transformation.

Step 2: Learning the Probability Distribution Using GAN

The transformed variable z is assumed to be sampled from an unknown distribution.

Before training:

The data is scaled to the range [-1, 1] using MinMaxScaler.

This scaling matches the Tanh activation of the generator output.

A Generative Adversarial Network (GAN) is trained using only samples of z.

Generator

Input: 10-dimensional Gaussian noise 
𝑁
(
0
,
1
)
N(0,1)

Output: 1-dimensional generated z sample

Discriminator

Input: Real or fake z sample

Output: Probability of being real

The networks are trained adversarially for 2000 epochs.

Step 3: PDF Approximation Using KDE

After training:

10,000 samples are generated using the trained generator.

Samples are inverse-scaled back to original range.

Kernel Density Estimation (KDE) is applied to:

Real transformed samples

GAN-generated samples

This produces a smooth estimate of the learned probability density function.

4. GAN Architecture
Generator
Layer	Description
Input	10-dimensional Gaussian noise
Dense	32 neurons, ReLU activation
Dense	32 neurons, ReLU activation
Output	1 neuron, Tanh activation
Discriminator
Layer	Description
Input	1-dimensional z value
Dense	32 neurons
Activation	LeakyReLU (α = 0.2)
Dense	16 neurons
Activation	LeakyReLU (α = 0.2)
Output	1 neuron, Sigmoid activation

The architecture is lightweight and appropriate for modeling a 1D distribution.

5. Results

The GAN successfully learns the probability density of the transformed variable.

The KDE plot shows:

The GAN-learned PDF closely follows the real transformed distribution.

The dominant peaks (modes) are captured.

The generated distribution is smooth and realistic.

Plot Included:

Real KDE vs GAN Learned KDE

Histogram of transformed variable z (before training)

6. Observations
Mode Coverage

The generator captures the dominant peaks of the transformed distribution without collapsing to a single mode.

Training Stability

Training remains stable over 2000 epochs.

Minor oscillations observed (normal in GAN training).

No severe mode collapse detected.

LeakyReLU in the discriminator improves gradient flow.

Quality of Generated Distribution

Generated samples produce a smooth probability density.

KDE of generated samples closely matches empirical KDE of real data.

Small deviations in low-density regions are expected.

7. Histogram

A histogram of the transformed variable z is generated before GAN training to visualize the empirical distribution.

8. Conclusion

A roll-number-based nonlinear transformation was applied to NO₂ concentration data.

A GAN was trained using only transformed samples to learn the unknown probability density function without assuming any analytical form.

The results demonstrate that GANs are effective in modeling unknown PDFs from real-world datasets using sample-based learning.

9. Software and Tools Used

Google Colab

Python

NumPy

Pandas

TensorFlow / Keras

Matplotlib

SciPy
