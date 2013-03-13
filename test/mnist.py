#!/usr/bin/env python

# Copyright (c) 2011 Leif Johnson <leif@leifjohnson.net>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import collections
import datetime
import glumpy
import logging
import numpy as np
import numpy.random as rng
import os
import pickle
import sys

import lmj.rbm
import idx_reader

FLAGS = argparse.ArgumentParser(
        conflict_handler='resolve',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
FLAGS.add_argument('--model', metavar='FILE',
                 help='load saved model from FILE')
FLAGS.add_argument('-i', '--images', metavar='FILE',
                 help='load image data from FILE')
FLAGS.add_argument('-l', '--labels', metavar='FILE',
                 help='load image labels from FILE')
FLAGS.add_argument('-g', '--gaussian', action='store_true',
                 help='use gaussian visible units')
FLAGS.add_argument('-c', '--conv', action='store_true',
                 help='use a convolutional network')
FLAGS.add_argument('-r', '--learning-rate', type=float, default=0.1, metavar='K',
                 help='use a learning rate of K')
FLAGS.add_argument('-m', '--momentum', type=float, default=0.2, metavar='K',
                 help='use a learning momentum of K')
FLAGS.add_argument('--l2', type=float, default=0.001, metavar='K',
                 help='apply L2 regularization with weight K')
FLAGS.add_argument('-p', '--sparsity', type=float, default=0.1, metavar='K',
                 help='set a target sparsity of K for hidden units')
FLAGS.add_argument('-n', '--n', type=int, default=10,
                 help='use NxN hidden units')
FLAGS.add_argument('-b', '--batch-size', type=int, default=257, metavar='N',
                 help='process N images in one minibatch')


if __name__ == '__main__':
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='%(levelname).1s %(asctime)s [%(module)s:%(lineno)d] %(message)s')

    args = FLAGS.parse_args()

    _visibles = np.zeros((args.n, 28, 28), dtype=np.float32)
    _hiddens = np.zeros((args.n, args.n), dtype=np.float32)
    _weights = np.zeros((args.n * args.n, 28, 28), dtype=np.float32)

    fig = glumpy.figure()

    visibles = [glumpy.image.Image(v) for v in _visibles]
    hiddens = glumpy.image.Image(_hiddens)
    weights = [glumpy.image.Image(w) for w in _weights]

    visible_frames = [
        fig.add_figure(args.n + 1, args.n, position=(args.n, r)).add_frame(aspect=1)
        for r in range(args.n)]

    weight_frames = [
        fig.add_figure(args.n + 1, args.n, position=(c, r)).add_frame(aspect=1)
        for r in range(args.n) for c in range(args.n)]

    loaded = False
    recent = collections.deque(maxlen=20)
    errors = [collections.deque(maxlen=20) for _ in range(10)]
    trainset = dict((i, []) for i in range(10))
    loader = idx_reader.iterimages(args.images, args.labels, True)

    Model = lmj.rbm.Convolutional if args.conv else lmj.rbm.RBM
    rbm = args.model and pickle.load(open(args.model, 'rb')) or Model(
        28 * 28, args.n * args.n, not args.gaussian)

    Trainer = lmj.rbm.ConvolutionalTrainer if args.conv else lmj.rbm.Trainer
    trainer = Trainer(rbm, l2=args.l2, momentum=args.momentum, target_sparsity=args.sparsity)

    def get_pixels():
        global loaded
        if not loaded and all(len(v) > 50 for v in trainset.itervalues()):
            loaded = True

        if loaded:
            t = rng.randint(10)
            pixels = trainset[t][rng.randint(len(trainset[t]))]
        else:
            t, pixels = loader.next()
            trainset[t].append(pixels)

        recent.append(pixels)
        if len(recent) < 20:
            raise RuntimeError

        return pixels

    def flatten(pixels):
        if not args.gaussian:
            return pixels.reshape((1, 28 * 28)) > 30.
        r = np.array(recent)
        mu = r.mean(axis=0)
        sigma = np.clip(r.std(axis=0), 0.1, np.inf)
        return ((pixels - mu) / sigma).reshape((1, 28 * 28))

    def unflatten(flat):
        if not args.gaussian:
            return 256. * flat.reshape((28, 28))
        r = np.array(recent)
        mu = r.mean(axis=0)
        sigma = r.std(axis=0)
        return sigma * flat.reshape((28, 28)) + mu

    def learn():
        batch = np.zeros((args.batch_size, 28 * 28), 'd')
        for i in range(args.batch_size):
            while True:
                try:
                    pixels = get_pixels()
                    break
                except RuntimeError:
                    pass
            flat = flatten(pixels)
            batch[i:i+1] = flat

        trainer.learn(batch, learning_rate=args.learning_rate)

        logging.debug('mean weight: %.3g, vis bias: %.3g, hid bias: %.3g',
                      rbm.weights.mean(), rbm.vis_bias.mean(), rbm.hid_bias.mean())

        return pixels, flat

    def update(pixels, flat):
        for i, (v, h) in enumerate(rbm.iter_passes(flat)):
            if i == len(visibles):
                break
            _visibles[i] = unflatten(v)
            [v.update() for v in visibles]

            _hiddens[:] = h.reshape((args.n, args.n))
            hiddens.update()

        for i in range(args.n * args.n):
            _weights[i] = rbm.weights[i].reshape((28, 28))
        [w.update() for w in weights]

        fig.redraw()

    @fig.event
    def on_draw():
        fig.clear(0, 0, 0, 0)
        for f in weight_frames + visible_frames:
            f.draw(x=f.x, y=f.y)
        for f, w in zip(weight_frames, weights):
            w.draw(x=f.x, y=f.y, z=0, width=f.width, height=f.height)
        for f, v in zip(visible_frames, visibles):
            v.draw(x=f.x, y=f.y, z=0, width=f.width, height=f.height)

    @fig.event
    def on_idle(dt):
        update(*learn())

    @fig.event
    def on_key_press(key, modifiers):
        if key == glumpy.window.key.ESCAPE:
            sys.exit()
        if key == glumpy.window.key.S:
            fn = datetime.datetime.now().strftime('rbm-%Y%m%d-%H%M%S.p')
            pickle.dump(rbm, open(fn, 'wb'))

    glumpy.show()
