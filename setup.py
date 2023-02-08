from setuptools import setup

setup(
    name='lsstseries',
    version='0.1.0',
    description='',
    url='https://github.com/lincc-frameworks/lsstseries',
    license='BSD 2-clause',
    install_requires=['pandas',
                      'numpy',
                      'dask',
                      'dask[distributed]',
                      'pyarrow',
                      'pyvo',
                      'scipy',
                      'coverage',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
