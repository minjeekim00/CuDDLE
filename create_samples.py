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


    def _get_dataset_obj(self):
        if self._dataset is None:
            self._dataset = dataset.load_dataset(**self._dataset_args)
        return self._dataset

    def _create_samples(self, num_gpus=1):
        # Set minibatch
        self.num_gpus = num_gpus
        self.minibatch_size = num_gpus * self.minibatch_per_gpu
        self._network_name = os.path.splitext(os.path.basename(self.network_pkl))[0]

        # Load inception
        with dnnlib.util.open_url('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metrics/inception_v3_features.pkl') as f: # identical to http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
            feature_net = pickle.load(f)

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
    
        # Generate real samples with batch
        save_dir = f'./datasets_mb/{self._dataset_name}_real'
        #if not os.path.exists(save_dir):
        self._create_reals(save_dir, max_reals=self.num_reals) #TODO: max_reals




        '''
        Gs_kwargs = {
        'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
        'randomize_noise': False
        }    
        label = np.zeros([1] + G.input_shapes[1][1:])
        os.makedirs('generated_cifar10', exist_ok=True)
        
        
        reals = self._real()
        
        
        real_embeddings = []
        print("Generating latent vector of real images from {}".format('./cifar-10-batches-py/data_batch_1'))
        
        for i in tqdm.tqdm(range(len(reals))):
            with tf.device('/gpu:0'):
#                 e = inception.get_output_for(tflib.convert_images_to_uint8(np.expand_dims(reals[i].reshape(32,32,3).transpose(2,0,1), axis=0)))
                e = inception.run(np.expand_dims(reals[i].reshape(32,32,3).transpose(2,0,1), axis=0))
#             real_embeddings.append(tflib.run(e)[0])
            real_embeddings.append(np.array(e).squeeze())
            
            
        print("Generating images from seed {} ...".format(42))
        fake_embeddings = []
        
        for i in tqdm.tqdm(range(self.num_fakes)):
            with tf.device('/gpu:0'):
                rnd = np.random.RandomState(42+i)
                z = rnd.randn(1, *G.input_shape[1:])
                image = G.run(z, label, **Gs_kwargs).astype(np.uint8).squeeze()
                im = Image.fromarray(image)
                im.save('generated_cifar10/generated_{:04d}.png'.format(i))
#                 e = inception.get_output_for(tflib.convert_images_to_uint8(np.expand_dims(image.transpose(2,0,1), axis=0)))
                e = inception.run(np.expand_dims(image.transpose(2,0,1), axis=0))
#             fake_embeddings.append(tflib.run(e)[0])
            fake_embeddings.append(np.array(e).squeeze())
            
        os.makedirs('latent_vectors', exist_ok=True)
        np.save('latent_vectors/'+config.network+'_real.npy', np.array(real_embeddings))
        np.save('latent_vectors/'+config.network+'_fake.npy', np.array(fake_embeddings))
        '''

    def _create_reals(self, save_dir, max_reals=None):
        print(f'Creating directory for real images in {save_dir}.....')
        start = time.time()
        dataset_obj = self._get_dataset_obj()
        os.makedirs(save_dir, exist_ok=True)

        num_real = 0
        while True:
            images_npy_path = os.path.join(save_dir, 'real_{:08d}_imgs.npy'.format(num_real))
            labels_npy_path = os.path.join(save_dir, 'real_{:08d}_lbls.npy'.format(num_real))
            if os.path.exists(images_npy_path):
                num = self.minibatch_size
            else:
                images = []
                labels = []
                for _ in range(self.minibatch_size):
                    image, label = dataset_obj.get_minibatch_np(1)
                    if image is None:
                        break
                    images.append(image)
                    labels.append(label)
                num = len(images)
                if num == 0:
                    break

                images = np.concatenate(images + [images[-1]] * (self.minibatch_size - num), axis=0)
                labels = np.concatenate(labels + [labels[-1]] * (self.minibatch_size - num), axis=0)
                np.save(images_npy_path, images)
                np.save(labels_npy_path, labels)

            if num < self.minibatch_size:
                break
            if max_reals is not None and num_real >= max_reals:
                break
            num_real += num
            print('\r%-20s%d' % ('Num real images:', num_real), end='', flush=True)

        print(f"\n Loading took {time.time()-start}")
        return


    def _real(self, path='./cifar-10-batches-py/data_batch_1'):
        with open(path, 'rb') as f:
            dict = pickle.load(f, encoding='bytes')
        return dict[b'data']
            
if __name__=='__main__':
    tflib.init_tf()

    seed = 1000
    network_pkl = "/home/minjee/projects/14.PublicDataset/pretrained/stylegan2/stylegan2-ffhq-config-f.pkl"
    tfrecord_dir = '/home/minjee/projects/14.PublicDataset/tfrecords/ffhq'
    num_gpus = 2

    config = config(seed, network_pkl, tfrecord_dir, num_reals=50000, num_fakes=50000)
    gen = sample_generator(config)
    gen._create_samples()
