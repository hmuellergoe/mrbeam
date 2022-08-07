import setuptools

setuptools.setup(
    name='MSI',
    version='0.1',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy>=1.14,<2.0',
        'scipy>=1.1,<2.0',
        'matplotlib',
        'lightwise',
    ],
    python_requires='>=3.6,<4.0',
)
