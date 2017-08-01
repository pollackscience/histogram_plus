from setuptools import setup

setup(name='histogram_plus',
      version='0.1.1',
      description='Overloaded histogram maker based on mpl',
      long_description='Documentation: https://brovercleveland.github.io/histogram_plus/',
      url='http://github.com/brovercleveland/histogram_plus',
      author='Brian Pollack',
      author_email='brianleepollack@gmail.com',
      license='MIT',
      packages=['histogram_plus'],
      install_requires=[ 'numpy', 'pandas', 'matplotlib'],
      zip_safe=False)
