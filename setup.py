from setuptools import setup, find_packages


if __name__ == '__main__':
    # Read in README.md for our long_description
    with open('./README.md', encoding='utf-8') as f:
        long_description = f.read()

    setup(
        name='torchdeq',
        version='0.1.0',
        license='MIT',
        author='Zhengyang Geng',
        author_email='zhengyanggeng@gmail.com',
        description='A PyTorch Lib for DEQs',
        url='https://github.com/locuslab/torchdeq',
        packages=find_packages(exclude=['deq-zoo', 'build', '*.egg-info']),
        long_description=long_description,
        long_description_content_type='text/markdown',
        classifiers=[
            'License :: OSI Approved :: MIT License',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Programming Language :: Python :: 3',
            'Operating System :: OS Independent',
            ],
        install_requires=[
            'torch>=1.11.0',
            'numpy>=1.21.5'
            ],
    )
