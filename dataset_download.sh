# # AFHQ Dataset
# git clone https://github.com/clovaai/stargan-v2.git
# cd stargan-v2/
# bash download.sh afhq-dataset

# # CIFAR-10 Dataset
# wget -P ./data https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
# tar -xvzf ./data/cifar-10-python.tar.gz
# rm ./data/cifar-10-python.tar.gz
# mkdir ./dataset/cifar10u

# # FFHQ Dataset
# pushd ~
# git clone https://github.com/NVlabs/ffhq-dataset.git
# cd ffhq-dataset
# python download_ffhq.py --tfrecords
# popd

# CelebA-HQ dataset
git clone https://github.com/suvojit-0x55aa/celebA-HQ-dataset-download.git
cd celebA-HQ-dataset-download
bash create_celebA-HQ.sh ./
