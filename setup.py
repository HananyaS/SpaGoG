import setuptools

setuptools.setup(
    name="SpaGoG",
    version="0.1",
    author="Shachar Hananya",
    author_email="shacharhananya@gmail.com",
    description="GoG models for sparse data clasification",
    license='MIT',
    packages=setuptools.find_packages(),
    package_data={
        'yamas': ['config.json']
    },
    install_requires=[
        'tqdm'
    ]
)
