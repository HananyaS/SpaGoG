import setuptools
from distutils.core import setup

setup(
    name='SpaGoG',  # How you named your package folder (MyLib)
    packages=setuptools.find_packages(),  # Chose the same as "name"
    version='0.21',  # Start with a small number and increase it with every change you make
    license='MIT',  # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    description='Sparse data classification using Graph of Graphs',  # Give a short description about your library
    author='Shachar Hananya',  # Type in your name
    author_email='shacharhananya@gmail.com',  # Type in your E-Mail
    # url='https://github.com/HananyaS/SpaGoG',  # Provide either the link to your github or to your website
    # download_url='https://github.com/HananyaS/SpaGoG/archive/refs/tags/v_0.2.tar.gz',  # I explain this later on
    keywords=['GoG', 'Missing values', 'Graphs'],  # Keywords that define your package best
    install_requires=[  # I get to this in a second
        'pandas',
        'torch',
        'torch-geometric',
        'matplotlib',
    ],
    # classifiers=[
    #     'License :: OSI Approved :: MIT License',  # Again, pick a license
    #     'Programming Language :: Python :: 3.8',
    # ],
    # add the json files that appear in default_params folder
    package_data={'spagog': ['default_params/*.json']},
)
