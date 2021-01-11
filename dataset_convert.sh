# AFHQ Dataset
mkdir datasets
cd datasets
mkdir afhqcat afhqdog afhqwild
cd ..

python dataset_tool.py create_from_images ./datasets/afhqcat ./stargan-v2/data/afhq/train/cat
python dataset_tool.py create_from_images ./datasets/afhqdog ./stargan-v2/data/afhq/train/dog
python dataset_tool.py create_from_images ./datasets/afhqwild ./stargan-v2/data/afhq/train/wild


# CIFAR-10 Dataset (Choose 1 option)
python dataset_tool.py create_cifar10 --ignore_labels=1 \
    ./datasets/cifar10u ./data/cifar-10-batches-py

python dataset_tool.py create_cifar10 --ignore_labels=0 \
    ~/datasets/cifar10c ~/downloads/cifar-10-batches-py

