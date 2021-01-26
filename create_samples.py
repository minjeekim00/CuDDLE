import tensorflow as tf
import pickle
import json
import argparse
import dnnlib
import dnnlib.tflib as tflib
import tqdm
import time
import numpy as np
import os
import sys
import glob
from PIL import Image

from training import dataset


#----------------------------------------------------------------------------

def error(msg):
    print('Error: ' + msg)
    exit(1)

#----------------------------------------------------------------------------

class config():
    def __init__(self, seed, network_pkl, tfrecord_dir, num_reals, num_fakes):
        self.seed         = seed
        self.network_pkl  = network_pkl
        self.tfrecord_dir = tfrecord_dir #TODO
        self.num_reals    = num_reals #TODO
        self.num_fakes    = num_fakes

#----------------------------------------------------------------------------

def _get_dataset_obj(dataset_args):
    return dataset.load_dataset(**dataset_args)

def _get_random_labels_tf(minibatch_size, isPggan=False): # => labels
    if isPggan:
        return tf.zeros([minibatch_size, 0], tf.float32)
    else:
        return tf.zeros([minibatch_size, 0], tf.int32)

# Choose dataset options
def _set_dataset_options(tfrecord_dir, Gs=None, seed=None, mirror=False): 
    dataset_options = dnnlib.EasyDict()
    dataset_options.path = tfrecord_dir
    dataset_options.mirror_augment = mirror
    dataset_options.seed = seed
    if Gs is not None:
        dataset_options.resolution = Gs.output_shapes[0][-1]
        dataset_options.max_label_size = Gs.input_shapes[1][-1]
    return dataset_options

def choose_shuffled_order(order): # Note: Images and labels must be added in shuffled order.
    np.random.RandomState(123).shuffle(order)
    return order

def make_data_path(outdir, dtype, idx):
    #types = ['imgs', 'lbls', 'img', 'feats']
    idx_str = f'{idx:08d}'

    img_path = os.path.join(outdir, f'{dtype}_{idx_str}_imgs.npy')
    lbl_path = os.path.join(outdir, f'{dtype}_{idx_str}_lbls.npy')
    png_path = os.path.join(outdir, f'{dtype}_{idx_str}_img.png')
    feat_path = os.path.join(outdir, f'{dtype}_{idx_str}_feats.npy')
    return img_path, lbl_path, png_path, feat_path

#---------------------------------------------------------------------------

def _create_reals_from_images(image_dir, save_dir, minibatch_size=8, max_reals=50000, seed=1000):
    print(f'Creating directory for real images in {save_dir}.....')
    start = time.time()

    image_filenames = sorted(glob.glob(os.path.join(image_dir, '*')))
    if len(image_filenames) == 0:
        error('No input images found')

    img = np.asarray(Image.open(image_filenames[0]))
    resolution = img.shape[0]
    channels = img.shape[2] if img.ndim == 3 else 1
    if img.shape[1] != resolution:
        error('Input images must have the same width and height')
    if resolution != 2 ** int(np.floor(np.log2(resolution))):
        error('Input image resolution must be a power-of-two')
    if channels not in [1, 3]:
        error('Input images must be stored as RGB or grayscale')

    order = choose_shuffled_order(np.arange(len(image_filenames)))
    num_real = 0

    while True:
        img_path, lbl_path, png_path, _ = make_data_path(save_dir, 'real', num_real)

        if os.path.exists(img_path) or os.path.exists(png_path):
            num = minibatch_size
        else:
            images = []
            labels = []
            for _ in range(minibatch_size):
                image = np.asarray(Image.open(image_filenames[order[num_real]]))
                #TODO: label
                if image is None:
                    break

                images.append(image)
                #labels.append(label)
            num = len(images)
            if num == 0:
                break

            images = np.concatenate(images + [images[-1]] * (minibatch_size - num), axis=0)
            #labels = np.concatenate(labels + [labels[-1]] * (minibatch_size - num), axis=0)

            if minibatch_size > 1:
                np.save(img_path, images)
                #np.save(lbl_path, labels)
            else:
                if images.shape[1] == 3:
                    image = np.transpose(images[0], (1,2,0))
                elif images.shape[1] == 1:
                    images = images[0][0]
                Image.fromarray(image).save(png_path)

        if num < minibatch_size:
            break
        if max_reals is not None and num_real >= max_reals:
            break
        num_real += num
        print('\r%-20s%d' % ('Num real images:', num_real), end='', flush=True)

    print(f"\n Creating took {time.time()-start}")
    return

#---------------------------------------------------------------------------

def _create_reals_from_tfrecord(tfrecord_dir, save_dir, minibatch_size=8, max_reals=50000, seed=1000):
    print(f'Creating directory for real images in {save_dir}.....')
    start = time.time()

    dataset_args = _set_dataset_options(tfrecord_dir, Gs=None, seed=seed)
    dataset_obj = _get_dataset_obj(dataset_args)
    os.makedirs(save_dir, exist_ok=True)

    num_real = 0
    while True:
        img_path, lbl_path, png_path, _ = make_data_path(save_dir, 'real', num_real)

        if os.path.exists(img_path) or os.path.exists(png_path):
            num = minibatch_size
        else:
            images = []
            labels = []
            for _ in range(minibatch_size):
                image, label = dataset_obj.get_minibatch_np(1)
                if image is None:
                    break
                images.append(image)
                labels.append(label)
            num = len(images)
            if num == 0:
                break

            images = np.concatenate(images + [images[-1]] * (minibatch_size - num), axis=0)
            labels = np.concatenate(labels + [labels[-1]] * (minibatch_size - num), axis=0)
            
            if minibatch_size > 1:
                np.save(img_path, images)
                #np.save(lbl_path, labels)
            else:
                if images.shape[1] == 3:
                    image = np.transpose(images[0], (1,2,0))
                elif images.shape[1] == 1:
                    images = images[0][0]
                Image.fromarray(image).save(png_path)

        if num < minibatch_size:
            break
        if max_reals is not None and num_real >= max_reals:
            break
        num_real += num
        print('\r%-20s%d' % ('Num real images:', num_real), end='', flush=True)

    print(f"\n Creating took {time.time()-start}")
    return

#----------------------------------------------------------------------------

def _create_real_embeddings(image_dir, save_dir, num_gpus=1, max_reals=50000, seed=1000):
    print(f'Creating directory for real images embeddings in {save_dir}.....')

    # Load inception
    with dnnlib.util.open_url('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metrics/inception_v3_features.pkl') as f: # identical to http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
        feature_net = pickle.load(f)

    start = time.time()
    os.makedirs(save_dir, exist_ok=True)

    num_real = 0
    
    while True:
        img_path, lbl_path, png_path, feat_path = make_data_path(image_dir, 'real', num_real)
        feat_path = feat_path.replace(image_dir, save_dir)

        if os.path.exists(img_path):
            images = np.load(img_path)
            #labels = np.load(lbl_path)
        elif os.path.exists(png_path):
            #TODO: grayscale
            images = np.asarray(Image.open(png_path))
            images = np.expand_dims(np.transpose(images, (2, 0, 1)), axis=0)
            
        if os.path.exists(feat_path):
            num_real += len(images)
            continue
        if max_reals is not None:
            #num = min(num, max_reals - num_real)
            num = max_reals - num_real
        if images.shape[1] == 1:
            images = np.tile(images, [1, 3, 1, 1])
        feats = feature_net.run(images, num_gpus=num_gpus, assume_frozen=True)[:num]

        np.save(feat_path, feats)
        #num_real += min(len(images), minibatch_size)
        print('\r%-20s%d' % ('Num real embeddings:', num_real), end='', flush=True)

        if max_reals is not None and num_real >= max_reals:
            break
        num_real += len(images)

#----------------------------------------------------------------------------

def _create_fakes(network_pkl, save_dir, save_emb_dir=None, 
<<<<<<< HEAD
        minibatch_per_gpu=8, num_gpus=1, G_kwargs=dict(is_validation=True), 
        max_fakes=50000, seed=1000, isPggan=False):
=======
        minibatch_per_gpu=8, num_gpus=1, G_kwargs=dict(is_validation=True), max_fakes=50000, seed=1000):
>>>>>>> 87b4980a12adbc3b0ce4a143618ea44081670f80

    start = time.time()

    minibatch_size = num_gpus * minibatch_per_gpu
    os.makedirs(save_dir, exist_ok=True)
    if save_emb_dir is not None:
        os.makedirs(save_emb_dir, exist_ok=True)
    
    # Load inception
    with dnnlib.util.open_url('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metrics/inception_v3_features.pkl') as f: # identical to http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
        feature_net = pickle.load(f)

    # Load network from pickle
    with dnnlib.util.open_url(network_pkl) as f:
        _G, _D, Gs = pickle.load(f)
        Gs.print_layers()

    # Construct TensorFlow graph.
    for gpu_idx in range(num_gpus):
        with tf.device('/gpu:%d' % gpu_idx):
            Gs_clone = Gs.clone()
            latents = tf.random_normal([minibatch_per_gpu] + Gs_clone.input_shape[1:])
            labels = _get_random_labels_tf(minibatch_per_gpu, isPggan)
            images = Gs_clone.get_output_for(latents, labels, **G_kwargs)
            if images.shape[1] == 1: images = tf.tile(images, [1, 3, 1, 1])
            images = tflib.convert_images_to_uint8(images)
            if save_emb_dir is not None:
                feature_net_clone = feature_net.clone()
                feat_fakes = feature_net_clone.get_output_for(images)

    num_fake = 0

    while True:
        img_path, lbl_path, png_path, feat_path = make_data_path(save_dir, 'fake', num_fake)
        ltnt_path = img_path.replace('_imgs', '_ltnts')
        feat_path = feat_path.replace(save_dir, save_emb_dir)

        if os.path.exists(img_path) or os.path.exists(png_path):
            num_fake += minibatch_size
            continue

        if minibatch_per_gpu > 1:
            np.save(img_path, tflib.run(images))
            #np.save(lbl_path, tflib.run(labels))
        else:
            image = tflib.run(images)
            if image.shape[1] == 3:
                image = np.transpose(image[0], (1,2,0))
            elif image.shape[1] == 1:
                image = image[0][0]
            Image.fromarray(image).save(png_path)
        
        np.save(ltnt_path, tflib.run(latents))
        if save_emb_dir is not None:
            np.save(feat_path, tflib.run(feat_fakes))

        if max_fakes is not None and num_fake >= max_fakes:
            break

        num_fake += minibatch_size
        print('\r%-20s%d' % ('Num fake images:', num_fake), end='', flush=True)


def execute_cmdline(argv):

    prog = argv[0]
    parser = argparse.ArgumentParser(
        prog        = prog,
        epilog      = 'Type "%s <command> -h" for more information.' % prog)

    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True

    def add_command(cmd, desc, example=None):
        epilog = 'Example: %s %s' % (prog, example) if example is not None else None
        return subparsers.add_parser(cmd, description=desc, help=desc, epilog=epilog)
    
    p = add_command(    '_create_reals_from_images', 'Create dataset from TFRecord archive.',
                        '_create_reals_from_images (image_dir) (save_dir) (minibatch_size). \
                        minibatch_size = 1 for png type')
    p.add_argument(     'image_dir',     help='TFRecord archive containing the images')
    p.add_argument(     'save_dir',         help='New dataset directory to be created')
    p.add_argument(     'minibatch_size',   help='Minibatch size to be saved. minibatch_size=1 for png save', type=int, default=8)
    #p.add_argument(     '--shuffle',        help='Randomize image order (default: 1)', type=int, default=1)

    p = add_command(    '_create_reals_from_tfrecord', 'Create dataset from TFRecord archive.',
                        '_create_reals_from_tfrecord (tfrecord_dir) (save_dir) (minibatch_size). \
                        minibatch_size = 1 for png type')
    p.add_argument(     'tfrecord_dir',     help='TFRecord archive containing the images')
    p.add_argument(     'save_dir',         help='New dataset directory to be created')
    p.add_argument(     'minibatch_size',   help='Minibatch size to be saved. minibatch_size=1 for png save', type=int, default=8)
    #p.add_argument(     '--shuffle',        help='Randomize image order (default: 1)', type=int, default=1)

    p = add_command(    '_create_real_embeddings', 'Create embedding vectors from real sample archive.',
                        '_create_real_embeddings (image_dir) (save_dir)')
    p.add_argument(     'image_dir',        help='Directory containing the images')
    p.add_argument(     'save_dir',         help='New dataset directory to be created')


    p = add_command(    '_create_fakes', 'Create embedding vectors from real sample archive.',
                        '_create_fakes (network_pkl) (save_dir) (save_emb_dir)')
    p.add_argument(     'network_pkl',      help='Directory containing the images')
    p.add_argument(     'save_dir',         help='New dataset directory to be created')
    p.add_argument(     'save_emb_dir',     help='New dataset directory to be created')
    p.add_argument(     'minibatch_per_gpu',help='Minibatch size to be saved. minibatch_size=1 for png save', type=int, default=8)
    p.add_argument(     'isPggan',          help='Is network PGGAN or not', type=bool, default=False)


    args = parser.parse_args(argv[1:] if len(argv) > 1 else ['-h'])
    func = globals()[args.command]
    del args.command
    func(**vars(args))



if __name__=='__main__':
    tflib.init_tf()
    execute_cmdline(sys.argv)
    
            
