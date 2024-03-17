import os
from distutils.core import setup
from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()
#    install_requires=required,

setup(
   name='uvr',
   version='0.1',
   description='Universal Voice Remover API',
   authors='Mohannad Barakat',
   author_email="Mohannad.Barakat@fau.de",
   license='MIT',
   package_dir={'uvr':'src'},
   long_description=open('README.md').read(),
   install_requires=required,
   url="https://github.com/NextAudioGen/ultimatevocalremover_api.git",
    package_data={
        'uvr': ['**/*.txt', '**/*.t7', '**/*.pth', '**/*.json', '**/*.yaml', '**/*.yml']
    }
)