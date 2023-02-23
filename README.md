# Official Code Implementation for CuDDLE
This is a repository for the paper [Barcode Method for Generative Model Evaluation driven by Topological Data Analysis](https://arxiv.org/abs/2106.02207).

[한국어 README](https://github.com/minjeekim00/Barcode/blob/main/README_KOR.md)

## Basic Concepts

- (Mutual) Fidelity : How well the generative model generates fake images.
- (Mutual) Diversity : How diverse the generated images are.

However, these two are relative values, not absolute values. Therefore, we suggest users to calculate **Relative Fidelity** and **Relative Diversity**.

- Relative Fidelity : (Fidelity between real and generated images) / (Fidelity between real and real images)
- Relative Diversity : (Diveristy between real and generated images)/(sqrt(Diveristy between real and real images) * sqrt(Diveristy between generated and generated images))

These two can be considered as normalized fidelity and normalized diversity.

## Usage

Take a look in

    barcode_example.py
    barcode.py

**barcode_example.py** is composed as:

```
from barcode import Barcode

superior = np.load('./brain_superior_embs.npz')['distance'].squeeze()
inferior = np.load('./brain_inferior_embs.npz')['distance'].squeeze()

barcode = Barcode(superior, inferior)
barcode_dict = barcode.get_barcode()
```

**barcode_example.py** imports **barcode.py**. Therefore, **barcode.py** should be contemplated as well.

If relative fidelity is close to 1, quality of the generated images as good as the original images.

If relative diversity is close to 1, the generative model generates images as diverse as the original images.


To plot barcode image, execute:

```
from barcode import Barcode

superior = np.load('./brain_superior_embs.npz')['distance'].squeeze()
inferior = np.load('./brain_inferior_embs.npz')['distance'].squeeze()

barcode = Barcode(superior, inferior)
barcode.plot_bars(mode='rf', title='Barcode plot between real and fake images', filename='realfake')
barcode.plot_bars(mode='rr', title='Barcode plot between real and real images', filename='realreal')
barcode.plot_bars(mode='ff', title='Barcode plot between fake and fake images', filename='fakefake')

```

## Environment

This code is written in tensorflow 1.15 version. However, the **barcode.py** code is executed in numpy ndarray format, therefore it does not depend on framework environment such as pytorch or tensorflow. All you have to do is:

1. Get embedding vectors from real and generated images using pretrained CNN network, such as [Inception v3](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf).
2. Transform embedded tensors to numpy ndarray format.
3. Calculate metrics following **barcode_example.py**

## Cite as

```
@article{jang2021barcode,
  title={Barcode Method for Generative Model Evaluation driven by Topological Data Analysis},
  author={Jang, Ryoungwoo and Kim, Minjee and Eun, Da-in and Cho, Kyungjin and Seo, Jiyeon and Kim, Namkug},
  journal={arXiv preprint arXiv:2106.02207},
  year={2021}
}
```
