# py-rbm

This is a small Python library that contains code for using and training
Restricted Boltzmann Machines (RBMs), the basic building blocks for many types
of deep belief networks. Variations available include the "standard" RBM (with
optional sparsity-based hidden layer learning); the temporal net introduced by
[Taylor, Hinton & Roweis][]; and convolutional nets with probabilistic
max-pooling described by [Lee, Grosse, Ranganath & Ng][].

Mostly I wrote the code to better understand the underlying algorithms. I don't
use it for anything at the moment, having moved on to using primarily [Theano][]
with [networks of rectified linear neurons][http://www.csri.utoronto.ca/~hinton/absps/reluICML.pdf]
(PDF). Still, there seems to be some interest in RBMs, so hopefully others will
find this package instructive, and maybe even useful !

[Taylor, Hinton & Roweis]: http://www.cs.nyu.edu/~gwtaylor/publications/nips2006mhmublv/
[Lee, Grosse, Ranganath & Ng]: http://cacm.acm.org/magazines/2011/10/131415-unsupervised-learning-of-hierarchical-representations-with-convolutional-deep-belief-networks/fulltext
[Theano]: http://deeplearning.net/software/theano/

## Installation

Just install using the included setup script :

    python setup.py install

Or you can install the package from the internets using pip :

    pip install lmj.rbm

## Testing

This library is definitely very alpha; so far I just have one main test that
encodes image data. To try things out, clone the source for this package and
install [glumpy][] :

    pip install glumpy

Then download the MNIST digits data from http://yann.lecun.com/exdb/mnist/ --
you'll need both the `train-*-images.ubyte.gz` and `train-*-labels.ubyte.gz`
files. Then run the test :

    python test/mnist.py \
      --images *-images.ubyte.gz \
      --labels *-labels.ubyte.gz

If you're feeling overconfident, go ahead and try out the gaussian visible
units :

    python test/mnist.py \
      --images *-images.ubyte.gz \
      --labels *-labels.ubyte.gz \
      --batch-size 257 \
      --l2 0.0001 \
      --learning-rate 0.2 \
      --momentum 0.5 \
      --sparsity 0.01 \
      --gaussian

The learning parameters can be a bit squirrely, but if things go right you
should see a number of images show up on your screen that represent the "basis
functions" that the network has learned when trying to auto-encode the MNIST
images you are feeding it.

You can also try running the test script with `--conv` to try out a
convolutional filterbank, but I'm not confident that the conv net test is
working correctly. Anyway, if you're thinking of using conv nets for a project,
please have a look at [Theano], or for a highly-tuned GPU/C++ implementation,
https://code.google.com/p/cuda-convnet/ (by
[Alex Krizhevsky][www.cs.toronto.edu/~kriz/]).

[glumpy]: http://code.google.com/p/glumpy/

## License

(The MIT License)

Copyright (c) 2011 Leif Johnson <leif@leifjohnson.net>

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the 'Software'), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
