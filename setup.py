from setuptools import setup, find_packages

setup(name='tfmodelzoo',
    version='0.1',
    description='Tool for easily loading pre-trained ImageNet models into TensorFlow via Keras Applications',
    url='https//github.com/joedav/tensorflow-modelzoo',
    author='Joe Davison',
    author_email='josephddavison@gmail.com',
    license='MIT',

    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',

        # Pick your license as you wish (should match "license" above)
         'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.

        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 2.7',
    ],
    keywords='tensorflow, deep learning, imagenet',
    install_requires=['tensorflow>=1.6'],
)
