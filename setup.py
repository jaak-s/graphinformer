import setuptools

exec(open("graphinformer/version.py").read())

setuptools.setup(
    name="graphinformer",
    version=__version__,
    author="Jaak Simm",
    author_email="jaak.simm@gmail.com",
    description="Graph Informer package",
    long_description="Pytorch implementation of Graph Informer",
    long_description_content_type="text/markdown",
    url="https://github.com/jaak-s/graphinformer",
    packages=setuptools.find_packages(),
    install_requires=["numpy", "torch", "tqdm", "scipy", "pandas"]
)
