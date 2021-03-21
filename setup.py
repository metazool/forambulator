from setuptools import setup

setup(
    name='forambulator',
    version='0.1.0',    
    description='Generate synthetic forams',
    url='https://github.com/metazool/forambulator',
    author='Jo Walsh',
    license='BSD 3-clause',
    packages=['forambulator'],
    install_requires=['numpy',
                      'requests',
                      'scikit-image',
                      'stylegan2-pytorch'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.5',
    ],
)

