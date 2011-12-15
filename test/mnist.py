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

import os
import sys
import time
import numpy
import glumpy
import pickle
import logging
import datetime
import optparse
import collections
import numpy.random as rng

from PIL import Image
from OpenGL import GL as gl

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
import lmj.rbm
import idx_reader

FLAGS = optparse.OptionParser()
FLAGS.add_option('', '--model', metavar='FILE',
                 help='load saved model from FILE')
FLAGS.add_option('-i', '--images', metavar='FILE',
                 help='load image data from FILE')
FLAGS.add_option('-l', '--labels', metavar='FILE',
                 help='load image labels from FILE')
FLAGS.add_option('-g', '--gaussian', action='store_true',
                 help='use gaussian visible units')
FLAGS.add_option('-r', '--learning-rate', type=float, default=0.1, metavar='K',
                 help='use a learning rate of K')
FLAGS.add_option('-m', '--momentum', type=float, default=0.2, metavar='K',
                 help='use a learning momentum of K')
FLAGS.add_option('', '--l2', type=float, default=0.001, metavar='K',
                 help='apply L2 regularization with weight K')
FLAGS.add_option('-p', '--sparsity', type=float, default=0.1, metavar='K',
                 help='set a target sparsity of K for hidden units')
FLAGS.add_option('-n', '--n', type=int, default=10,
                 help='use NxN hidden units')
FLAGS.add_option('-b', '--batch-size', type=int, default=5, metavar='N',
                 help='process N images in one minibatch')


if __name__ == '__main__':
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='%(levelname).1s %(asctime)s [%(module)s:%(lineno)d] %(message)s')

    opts, args = FLAGS.parse_args()

    _visibles = numpy.zeros((opts.n, 28, 28), dtype=numpy.float32)
    _hiddens = numpy.zeros((opts.n, opts.n), dtype=numpy.float32)
    _weights = numpy.zeros((opts.n * opts.n, 28, 28), dtype=numpy.float32)

    fig = glumpy.figure()

    m = opts.gaussian and 0.5 or 1.5
    visibles = [glumpy.image.Image(v) for v in _visibles]
    hiddens = glumpy.image.Image(_hiddens)
    weights = [glumpy.image.Image(w, vmin=-m, vmax=m) for w in _weights]

    visible_frames = [
        fig.add_figure(opts.n + 1, opts.n, position=(opts.n, r)).add_frame(aspect=1)
        for r in range(opts.n)]

    weight_frames = [
        fig.add_figure(opts.n + 1, opts.n, position=(c, r)).add_frame(aspect=1)
        for r in range(opts.n) for c in range(opts.n)]

    loaded = False
    updates = -1
    recent = collections.deque(maxlen=20)
    errors = [collections.deque(maxlen=20) for _ in range(10)]
    testset = [None] * 10
    trainset = dict((i, []) for i in range(10))
    loader = idx_reader.iterimages(opts.labels, opts.images, False)

    rbm = opts.model and pickle.load(open(opts.model, 'rb')) or lmj.rbm.RBM(
        28 * 28, opts.n * opts.n, not opts.gaussian)

    trainer = lmj.rbm.Trainer(
        rbm, l2=opts.l2, momentum=opts.momentum, target_sparsity=opts.sparsity)

    def get_pixels():
        global loaded
        if not loaded and numpy.all([len(trainset[t]) > 50 for t in range(10)]):
            loaded = True

        if loaded:
            t = rng.randint(10)
            pixels = trainset[t][rng.randint(len(trainset[t]))]
        else:
            t, pixels = loader.next()
            if testset[t] is None and rng.random() < 0.3:
                testset[t] = pixels
                raise RuntimeError
            else:
                trainset[t].append(pixels)

        recent.append(pixels)
        if len(recent) < 20:
            raise RuntimeError

        return pixels

    def flatten(pixels):
        if not opts.gaussian:
            return pixels.reshape((1, 28 * 28)) > 30.
        r = numpy.array(recent)
        mu = r.mean(axis=0)
        sigma = numpy.clip(r.std(axis=0), 0.1, numpy.inf)
        return ((pixels - mu) / sigma).reshape((1, 28 * 28))

    def unflatten(flat):
        if not opts.gaussian:
            return 256. * flat.reshape((28, 28))
        r = numpy.array(recent)
        mu = r.mean(axis=0)
        sigma = r.std(axis=0)
        return sigma * flat.reshape((28, 28)) + mu

    def learn():
        batch = numpy.zeros((opts.batch_size, 28 * 28), 'd')
        for i in range(opts.batch_size):
            while True:
                try:
                    pixels = get_pixels()
                    break
                except RuntimeError:
                    pass
            flat = flatten(pixels)
            batch[i:i+1] = flat

        trainer.learn(batch, learning_rate=opts.learning_rate)

        logging.debug('mean weight: %.3g, vis bias: %.3g, hid bias: %.3g',
                      rbm.weights.mean(), rbm.vis_bias.mean(), rbm.hid_bias.mean())

        return pixels, flat

    def update(pixels, flat):
        for i, (v, h) in enumerate(rbm.iter_passes(flat)):
            if i == len(visibles):
                break
            _visibles[i] = unflatten(v)
            [v.update() for v in visibles]

            _hiddens[:] = h.reshape((opts.n, opts.n))
            hiddens.update()

        for i in range(opts.n * opts.n):
            _weights[i] = rbm.weights[i].reshape((28, 28))
        [w.update() for w in weights]

        fig.redraw()

    def evaluate():
        for t, pixels in enumerate(testset):
            if pixels is None:
                continue
            estimate = unflatten(rbm.reconstruct(flatten(pixels)))
            errors[t].append(((pixels - estimate) ** 2).mean())
        def mean(x):
            return sum(x) / max(1, len(x))
        logging.error(' : '.join('%d' % mean(errors[t]) for t in range(10)))

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
        global updates
        if updates:
            updates -= 1
            update(*learn())
            evaluate()

    @fig.event
    def on_key_press(key, modifiers):
        global updates
        if key == glumpy.window.key.ESCAPE:
            sys.exit()
        if key == glumpy.window.key.S:
            fn = datetime.datetime.now().strftime('rbm-%Y%m%d-%H%M%S.p')
            pickle.dump(rbm, open(fn, 'wb'))
        if key == glumpy.window.key.SPACE:
            updates = updates == 0 and -1 or 0
        if key == glumpy.window.key.ENTER:
            if updates >= 0:
                updates = 1

    glumpy.show()
