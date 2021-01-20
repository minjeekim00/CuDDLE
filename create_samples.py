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
from PIL import Image

from training import dataset

class config():
    def __init__(self, seed, network_pkl, tfrecord_dir, num_reals, num_fakes):
        self.seed         = seed
        self.network_pkl  = network_pkl
        self.tfrecord_dir = tfrecord_dir #TODO
        self.num_reals    = num_reals #TODO
        self.num_fakes    = num_fakes

        self._dataset_args  = dnnlib.EasyDict()

class sample_generator():
    def __init__(self, config):
        self.seed         = config.seed
        self.num_reals    = config.num_reals
        self.num_fakes    = config.num_fakes
        self.tfrecord_dir = config.tfrecord_dir
        self.network_pkl  = config.network_pkl

        self._dataset_args  = config._dataset_args

        self.shuffle        = False
        self.max_images     = None
        self.run_dir        = None
        self.num_gpus       = 1
        self.minibatch_per_gpu = 8 #TODO

        self._network_name  = ''
        self._dataset       = None
        self._dataset_name  = ''
        self.feature_net    = None


    def _get_dataset_obj(self):
        if self._dataset is None:
            self._dataset = dataset.load_dataset(**self._dataset_args)
        return self._dataset

    def _get_random_labels_tf(self, minibatch_size):
        return self._get_dataset_obj().get_random_labels_tf(minibatch_size)

    def _create_samples(self, num_gpus=1, as_image=False):
        # Set minibatch
        self.num_gpus = num_gpus
        self.minibatch_size = num_gpus * self.minibatch_per_gpu
        self._network_name = os.path.splitext(os.path.basename(self.network_pkl))[0]

        # Load inception
        with dnnlib.util.open_url('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metrics/inception_v3_features.pkl') as f: # identical to http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
            self.feature_net = pickle.load(f)

        # Load network from pickle
        with dnnlib.util.open_url(self.network_pkl) as f:
            _G, _D, Gs = pickle.load(f)
            Gs.print_layers()

        # Look up training options.
        self.run_dir = os.path.dirname(self.network_pkl) #TODO: potential run_dir
        training_options = None

        # Choose dataset options 
        dataset_options = dnnlib.EasyDict()
        dataset_options.path = self.tfrecord_dir
        dataset_options.mirror_augment = False #TODO
        dataset_options.resolution = Gs.output_shapes[0][-1]
        dataset_options.max_label_size = Gs.input_shapes[1][-1]
        dataset_options.seed = self.seed
        self._dataset_args = dataset_options
        self._dataset_name = os.path.splitext(os.path.basename(self.tfrecord_dir))[0]
            
        print()
        print('Dataset options:')
        print(json.dumps(dataset_options, indent=2))
    
        # Generate fake samples (minibatch)
        save_dir = f'./datasets_mb/{self._dataset_name}_fake'
        if not os.path.exists(save_dir):
        G_kwargs=dict(is_validation=True)
        self._create_fakes(Gs, G_kwargs, save_dir, w_embeddings=True, max_reals=None)

    def _create_fakes(self, Gs, G_kwargs, save_dir, w_embeddings=True, max_reals=None):
        os.makedirs(save_dir, exist_ok=True)
        num_gpus = self.num_gpus

        if w_embeddings:
            save_dir_emb=save_dir+'_emb_incpt'
            os.makedirs(save_dir_emb, exist_ok=True)

        # Construct TensorFlow graph.
        for gpu_idx in range(num_gpus):
            with tf.device('/gpu:%d' % gpu_idx):
                Gs_clone = Gs.clone()
                latents = tf.random_normal([self.minibatch_per_gpu] + Gs_clone.input_shape[1:])
                labels = self._get_random_labels_tf(self.minibatch_per_gpu)
                images = Gs_clone.get_output_for(latents, labels, **G_kwargs)
                if images.shape[1] == 1: images = tf.tile(images, [1, 3, 1, 1])
                images = tflib.convert_images_to_uint8(images)
                if w_embeddings:
                    feature_net_clone = self.feature_net.clone()
                    feat_fakes = feature_net_clone.get_output_for(images)

        max_fakes = self.num_fakes
        num_fake = 0

        while True:
            images_npy_path = os.path.join(save_dir, 'fake_{:08d}_imgs.npy'.format(num_fake))
            labels_npy_path = os.path.join(save_dir, 'fake_{:08d}_lbls.npy'.format(num_fake))
            latents_npy_path = os.path.join(save_dir, 'fake_{:08d}_ltnts.npy'.format(num_fake))
            feats_npy_path = os.path.join(save_dir_emb, 'fake_{:08d}_feats.npy'.format(num_fake))

            if os.path.exists(images_npy_path):
                num_fake += self.minibatch_size
                continue

            np.save(images_npy_path, images.eval())
            #np.save(labels_npy_path, labels.eval())
            np.save(latents_npy_path, latents.eval())
            if w_embeddings:
                np.save(feats_npy_path, feat_fakes.eval())

            if max_fakes is not None and num_fake >= max_fakes:
                break

            num_fake += self.minibatch_size
            print('\r%-20s%d' % ('Num fake images:', num_fake), end='', flush=True)

        
def _get_dataset_obj(dataset_args):
    return dataset.load_dataset(**dataset_args)

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

#-----------------------------------------------------------------------------------
def _create_reals_from_tfrecord(tfrecord_dir, save_dir, minibatch_size=8, max_reals=50000, seed=1000):
    print(f'Creating directory for real images in {save_dir}.....')
    start = time.time()

    dataset_args = _set_dataset_options(tfrecord_dir, Gs=None, seed=seed)
    dataset_obj = _get_dataset_obj(dataset_args)
    os.makedirs(save_dir, exist_ok=True)

    num_real = 0
    while True:
        images_npy_path = os.path.join(save_dir, 'real_{:08d}_imgs.npy'.format(num_real))
        labels_npy_path = os.path.join(save_dir, 'real_{:08d}_lbls.npy'.format(num_real))
        image_png_path  = os.path.join(save_dir, 'real_{:08d}_img.png'.format(num_real))

        if os.path.exists(images_npy_path) or os.path.exists(image_png_path):
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
                np.save(images_npy_path, images)
                #np.save(labels_npy_path, labels)
            else:
                if images.shape[1] == 3:
                    image = np.transpose(images[0], (1,2,0))
                elif images.shape[1] == 1:
                    images = images[0][0]
                Image.fromarray(image).save(image_png_path)

        if num < minibatch_size:
            break
        if max_reals is not None and num_real >= max_reals:
            break
        num_real += num
        print('\r%-20s%d' % ('Num real images:', num_real), end='', flush=True)

    print(f"\n Creating took {time.time()-start}")
    return

#-----------------------------------------------------------------------------------

def _create_real_embeddings(image_dir, save_dir, num_gpus=1, max_reals=50000, seed=1000):
    print(f'Creating directory for real images embeddings in {save_dir}.....')

    # Load inception
    with dnnlib.util.open_url('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metrics/inception_v3_features.pkl') as f: # identical to http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
        feature_net = pickle.load(f)

    start = time.time()
    os.makedirs(save_dir, exist_ok=True)

    num_real = 0
    
    while True:
        images_npy_path = os.path.join(image_dir, 'real_{:08d}_imgs.npy'.format(num_real))
        labels_npy_path = os.path.join(image_dir, 'real_{:08d}_lbls.npy'.format(num_real))
        image_png_path  = os.path.join(image_dir, 'real_{:08d}_img.png'.format(num_real))

        if os.path.exists(images_npy_path):
            images = np.load(images_npy_path)
            #labels = np.load(labels_npy_path)
        elif os.path.exists(image_png_path):
            #TODO: grayscale
            try:
                images = np.asarray(Image.open(image_png_path))
                images = np.expand_dims(np.transpose(images, (2, 0, 1)), axis=0)
            except:
                print(image_png_path)
                print(np.asarray(Image.open(image_png_path)).shape)
            
        feat_npy_path = os.path.join(save_dir, 'real_{:08d}_feats.npy'.format(num_real))

        if os.path.exists(feat_npy_path):
            num_real += len(images)
            continue
        if max_reals is not None:
            #num = min(num, max_reals - num_real)
            num = max_reals - num_real
        if images.shape[1] == 1:
            images = np.tile(images, [1, 3, 1, 1])
        feats = feature_net.run(images, num_gpus=num_gpus, assume_frozen=True)[:num]

        np.save(feat_npy_path, feats)
        #num_real += min(len(images), minibatch_size)
        print('\r%-20s%d' % ('Num real embeddings:', num_real), end='', flush=True)

        if max_reals is not None and num_real >= max_reals:
            break
        num_real += len(images)

#-----------------------------------------------------------------------------------


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

    args = parser.parse_args(argv[1:] if len(argv) > 1 else ['-h'])
    func = globals()[args.command]
    del args.command
    func(**vars(args))



if __name__=='__main__':
    tflib.init_tf()
    execute_cmdline(sys.argv)
    
            
#if __name__=='__main__':
    #tflib.init_tf()

    #seed = 1000
    #num_gpus = 1
    #network_pkl = "/home/minjee/projects/14.PublicDataset/pretrained/stylegan2/stylegan2-ffhq-config-f.pkl"
    #tfrecord_dir = '/home/minjee/projects/14.PublicDataset/tfrecords/ffhq'
    #network_pkl = '/home/minjee/projects/14.PublicDataset/pretrained/stylegan/karras2019stylegan-celebahq-1024x1024.pkl'
    #tfrecord_dir = '/home/minjee/projects/14.PublicDataset/tfrecords/celebahq'

    #network_pkl = '/home/minjee/projects/02.StyleGAN2/brainct/brainct_8bit_normal_rgb_results/00001-stylegan2-brain_ct_normal_tfr-2gpu-config-f/network-snapshot-014438.pkl'
    #tfrecord_dir = '/home/minjee/projects/01.PGGAN/brainct/brainct_8bit_normal_rgb_tfr'

    #config = config(seed, network_pkl, tfrecord_dir, num_reals=50000, num_fakes=50000)
    #gen = sample_generator(config)
    #gen._create_samples()
