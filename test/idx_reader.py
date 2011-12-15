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

'''A Python library for reading MNIST handwritten digit database (IDX) files.'''

import gzip
import numpy
import struct


def iterimages(label_file, image_file, gzip=False):
    '''Iterate over labels and images from the MNIST handwritten digit dataset.

    This function generates (label, image) pairs, one for each image in the
    dataset. The image is represented as a numpy array of the pixel values, and
    the label is an integer.

    label_file: The name of a binary IDX file to load label data from.
    image_file: The name of a binary IDX file to load image data from.
    gzip: If True, the binary files will be gunzipped automatically before
      reading.
    '''
    opener = gzip and gzip.open or open

    # check the label header
    handle = opener(label_file, 'rb')
    label_data = handle.read()
    handle.close()
    magic, label_count = struct.unpack('>2i', label_data[:8])
    assert magic == 2049
    assert label_count > 0
    label_data = label_data[8:]

    # check the image header
    handle = opener(image_file, 'rb')
    image_data = handle.read()
    handle.close()
    magic, image_count, rows, columns = struct.unpack('>4i', image_data[:16])
    assert magic == 2051
    assert image_count > 0
    assert rows > 0
    assert columns > 0
    image_data = image_data[16:]

    # check that the two files agree on cardinality
    assert image_count == label_count

    for _ in range(image_count):
        label, = struct.unpack('B', label_data[:1])
        label_data = label_data[1:]

        count = rows * columns
        pixels = struct.unpack('%dB' % count, image_data[:count])
        image_data = image_data[count:]

        yield label, numpy.array(pixels).astype(float).reshape((rows, columns))


if __name__ == '__main__':
    import sys
    import glumpy

    iterator = iterimages(sys.argv[1], sys.argv[2], False)
    composites = [numpy.zeros((28, 28), 'f') for _ in range(10)]
    images = [glumpy.Image(c) for c in composites]

    win = glumpy.Window(800, 600)

    @win.event
    def on_draw():
        win.clear()
        w, h = win.get_size()
        for i, image in enumerate(images):
            image.blit(w * (i % 5) / 5., h * (i // 5) / 2., w / 5., h / 2.)

    @win.event
    def on_idle(dt):
        try:
            label, pixels = iterator.next()
        except StopIteration:
            sys.exit()
        composites[label] *= 0.3
        composites[label] += 0.7 * pixels
        images[label].update()
        win.draw()

    win.mainloop()
