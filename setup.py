from distutils.core import setup
setup(
  name = 'SpaGoG',         # How you named your package folder (MyLib)
  packages = ['SpaGoG'],   # Chose the same as "name"
  version = '0.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Sparse data classification using Graph of Graphs models',   # Give a short description about your library
  author = 'Shachar Hananya',                   # Type in your name
  author_email = 'shacharhananya@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/HananyaS/SpaGoG',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/HananyaS/SpaGoG/archive/refs/tags/v_0.1.tar.gz',    # I explain this later on
  keywords = ["GoG"],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'numpy',
          'pandas',
          'torch',
          'torch_geometric',
      ],
)