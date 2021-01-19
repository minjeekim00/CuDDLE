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
        self.feature_net    = None


    def _get_dataset_obj(self):
        if self._dataset is None:
            self._dataset = dataset.load_dataset(**self._dataset_args)
        return self._dataset

    def _get_random_labels_tf(self, minibatch_size):
        return self._get_dataset_obj().get_random_labels_tf(minibatch_size)

    def _create_samples(self, num_gpus=1):
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
    
        # Generate real samples (minibatch)
        save_dir = f'./datasets_mb/{self._dataset_name}_real'
        if not os.path.exists(save_dir):
            self._create_reals(save_dir, max_reals=self.num_reals) #TODO: max_reals
        
        # Generate real sample embeddings (minibatch)
        save_dir = f'./datasets_mb/{self._dataset_name}_real_emb_incpt'
        if not os.path.exists(save_dir):
            self._create_real_embeddings(save_dir, max_reals=self.num_reals)

        # Generate fake samples (minibatch)
        save_dir = f'./datasets_mb/{self._dataset_name}_fake'
        #if not os.path.exists(save_dir):
        G_kwargs=dict(is_validation=True)
        self._create_fakes(Gs, G_kwargs, save_dir, w_embeddings=True, max_reals=None)




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
                #np.save(labels_npy_path, labels)

            if num < self.minibatch_size:
                break
            if max_reals is not None and num_real >= max_reals:
                break
            num_real += num
            print('\r%-20s%d' % ('Num real images:', num_real), end='', flush=True)

        print(f"\n Creating/Loading took {time.time()-start}")
        return

    def _get_real_batch(self, save_dir, num_real=None):
        assert os.path.exists(save_dir)

        if not os.path.exists(save_dir):
            self._create_reals(save_dir, max_reals)

        images_npy_path = os.path.join(save_dir, 'real_{:08d}_imgs.npy'.format(num_real))
        labels_npy_path = os.path.join(save_dir, 'real_{:08d}_lbls.npy'.format(num_real))

        images = np.load(images_npy_path)
        #labels = np.load(labels_npy_path)
        num = len(images)
        #return images, labels, num
        return images, _, num


    def _create_real_embeddings(self, save_dir, max_reals=None):
        print(f'Creating directory for real images embeddings in {save_dir}.....')
        start = time.time()
        dataset_obj = self._get_dataset_obj()
        minibatch_size = self.minibatch_size
        os.makedirs(save_dir, exist_ok=True)

        num_real = 0
        while True:
            print("num_real", num_real)
            images, _labels, num = self._get_real_batch(save_dir[:len('_emb_incpt') * -1], num_real)
            feat_npy_path = os.path.join(save_dir, 'real_{:08d}_feats.npy'.format(num_real))

            if os.path.exists(feat_npy_path):
                num_real += min(len(images), minibatch_size)
                continue
            if self.max_reals is not None:
                num = min(num, max_reals - num_real)
            if images.shape[1] == 1:
                images = np.tile(images, [1, 3, 1, 1])
            feats = self.feature_net.run(images, num_gpus=self.num_gpus, assume_frozen=True)[:num]

            np.save(feat_npy_path, feats)

            num_real += min(len(images), minibatch_size)
            print('\r%-20s%d' % ('Num real embeddings:', num_real), end='', flush=True)

            if max_reals is not None and num_real >= max_reals:
                break

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

            if os.path.exists():

            np.save(images_npy_path, images.eval())
            #np.save(labels_npy_path, labels.eval())
            np.save(latents_npy_path, latents.eval())
            if w_embeddings:
                np.save(feats_npy_path, feat_fakes.eval())

            if max_fakes is not None and num_fake >= max_fakes:
                break

            num_fake += self.minibatch_size
            print('\r%-20s%d' % ('Num fake images:', num_fake), end='', flush=True)

        
        '''
        # Calculate statistics for fakes.
        start = time.time()
        feat_fake = []
        for begin in range(0, self.num_fakes, minibatch_size):
            self._report_progress(begin, self.num_fakes)
            feat_fake += list(np.concatenate(tflib.run(result_expr), axis=0))
        feat_fake = np.stack(feat_fake[:self.num_fakes])
        mu_fake = np.mean(feat_fake, axis=0)
        sigma_fake = np.cov(feat_fake, rowvar=False)
        print(f"Calculating fake statistics took {time.time()-start} secs...")
        '''

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
