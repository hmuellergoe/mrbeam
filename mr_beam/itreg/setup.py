import setuptools

setuptools.setup(
    name='regpy',
    version='0.2',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy>=1.14,<2.0',
        'scipy>=1.1,<2.0',
    ],
    extras_require={
        'nfft': [
            'pyNFFT>=1.3,<2.0',
        ],
    },
    python_requires='>=3.6,<4.0',
)
