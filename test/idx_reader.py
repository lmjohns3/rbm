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
import numpy as np
import struct


def iterimages(image_file, label_file=None, unzip=False):
    '''Iterate over labels and images from the MNIST handwritten digit dataset.

    This function generates (label, image) pairs, one for each image in the
    dataset. The image is represented as a numpy array of the pixel values, and
    the label is an integer.

    image_file: The name of a binary IDX file to load image data from.
    label_file: The name of a binary IDX file to load label data from.
    ungzip: If True, the binary files will be gunzipped automatically before
      reading.
    '''
    opener = unzip and gzip.open or open

    # check the label header
    label_count = None
    if label_file:
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
    assert label_count is None or image_count == label_count

    for _ in range(image_count):
        label = None
        if label_count:
            label, = struct.unpack('B', label_data[:1])
            label_data = label_data[1:]

        count = rows * columns
        pixels = struct.unpack('%dB' % count, image_data[:count])
        image_data = image_data[count:]

        yield label, np.array(pixels).astype(float).reshape((rows, columns))


if __name__ == '__main__':
    import sys
    import glumpy

    iterator = iterimages(sys.argv[1], sys.argv[2], False)
    composites = [np.zeros((28, 28), 'f') for _ in range(10)]

    fig = glumpy.Figure()
    images_and_frames = []
    for i, c in enumerate(composites):
        frame = fig.add_figure(rows=2, cols=5, position=divmod(i, 2)).add_frame(aspect=1)
        images_and_frames.append((glumpy.Image(c), frame))

    @fig.event
    def on_draw():
        fig.clear()
        for image, frame in images_and_frames:
            image.update()
            frame.draw(x=frame.x, y=frame.y)
            image.draw(x=frame.x, y=frame.y, z=0, width=frame.width, height=frame.height)

    @fig.event
    def on_idle(dt):
        try:
            label, pixels = iterator.next()
        except StopIteration:
            sys.exit()
        composites[label] *= 0.3
        composites[label] += 0.7 * pixels
        fig.redraw()

    glumpy.show()
