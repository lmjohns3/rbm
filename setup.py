import setuptools

setuptools.setup(
    name='lmj.rbm',
    version='0.1',
    py_modules=['lmj.rbm'],
    namespace_packages=['lmj'],
    install_requires=['numpy'],
    author='Leif Johnson',
    author_email='leif@leifjohnson.net',
    description='A library of Restricted Boltzmann Machines',
    long_description=open('README.md').read(),
    license='MIT',
    keywords=('deep-belief-network '
              'restricted-boltzmann-machine '
              'machine-learning'),
    url='http://github.com/lmjohns3/py-rbm/',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
    )
