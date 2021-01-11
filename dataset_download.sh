# AFHQ Dataset
git clone https://github.com/clovaai/stargan-v2.git
cd stargan-v2/
bash download.sh afhq-dataset

# CIFAR-10 Dataset
wget -P ./data https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvzf ./data/cifar-10-python.tar.gz
rm ./data/cifar-10-python.tar.gz
mkdir ./dataset/cifar10u
