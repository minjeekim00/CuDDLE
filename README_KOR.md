# Barcode 공식 구현체
이 레포지토리는 [Barcode Method for Generative Model Evaluation driven by Topological Data Analysis](https://arxiv.org/abs/2106.02207) 논문의 공식 구현체입니다.

## 기본 개념

- (Mutual) Fidelity : 생성 모델이 얼마나 가짜 이미지를 잘 생성하는지를 나타내주는 지표.
- (Mutual) Diversity : 생성된 이미지가 얼마나 다양한지를 나타내주는 지표.

하지만 이들은 절대적 수치가 아니라 상대적 수치고, 따라서 보정값이 필요합니다. 따라서 **Relative Fidelity**와 **Relative Diversity**를 계산하는 것을 권장합니다.

- Relative Fidelity : (진짜 이미지와 가짜 이미지 사이의 Fidelity) / (진짜와 진짜 이미지 사이의 Fidelity)
- Relative Diversity : (진짜와 가짜 이미지 사이의 Diversity)/(sqrt(진짜와 진짜 이미지 사이의 Diveristy) * sqrt(가짜와 가짜 이미지 사이의 Diveristy))

이들은 정규화된 Fidelity와 Diversity로 간주할 수 있습니다.

## 사용법

    barcode_example.py
    barcode.py
    
를 참고하세요.

**barcode_example.py** 는 다음처럼 구성되어 있습니다:

```
from barcode import Barcode

superior = np.load('./brain_superior_embs.npz')['distance'].squeeze()
inferior = np.load('./brain_inferior_embs.npz')['distance'].squeeze()

barcode = Barcode(superior, inferior)
barcode_dict = barcode.get_barcode()
```

**barcode_example.py** 는 **barcode.py** 를 호출합니다. 따라서, **barcode.py** 또한 들여다 보는 것을 추천합니다.

Relative fidelity가 1에 가깝다면 생성된 이미지는 원래 이미지만큼 퀄리티가 좋습니다.

Relative diversity가 1에 가깝다면 생성된 이미지는 원래 이미지만큼 다양한 이미지를 포함하고 있습니다.

Barcode 이미지를 생성하려면 다음 코드를 실행하세요:

```
from barcode import Barcode

superior = np.load('./brain_superior_embs.npz')['distance'].squeeze()
inferior = np.load('./brain_inferior_embs.npz')['distance'].squeeze()

barcode = Barcode(superior, inferior)
barcode.plot_bars(mode='rf', title='Barcode plot between real and fake images', filename='realfake')
barcode.plot_bars(mode='rr', title='Barcode plot between real and real images', filename='realreal')
barcode.plot_bars(mode='ff', title='Barcode plot between fake and fake images', filename='fakefake')

```

## 구동 환경

이 코드는 텐서플로우 1.15 버전으로 쓰여있지만 내부 알고리즘은 Numpy의 ndarray위에서 돌아가기 때문에 프레임워크에 구애받지 않습니다. 사용자가 할 일은:

1. 사전 학습된 [Inception v3](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)와 같은 CNN모델을 통해 embedding vector를 얻습니다.
2. Embedded vector들을 numpy ndarray format으로 변환합니다.
3. **barcode_example.py** 를 통해 수치를 계산합니다.
## 인용

```
@article{jang2021barcode,
  title={Barcode Method for Generative Model Evaluation driven by Topological Data Analysis},
  author={Jang, Ryoungwoo and Kim, Minjee and Eun, Da-in and Cho, Kyungjin and Seo, Jiyeon and Kim, Namkug},
  journal={arXiv preprint arXiv:2106.02207},
  year={2021}
}
```
