from setuptools import setup, find_packages


with open("requirements.txt", "r") as fh:
    requirements = fh.read().splitlines()

setup(
    name="recsyslib",
    version="0.0.1",
    use_scm_version=True,
    url="https://github.com/TimSchmeier/recsyslib",
    packages=find_packages(exclude=("tests")),
    include_package_data=True,
    setup_requires=["setuptools_scm"],
    tests_require=["pytest"],
    install_requires=[requirements],
    classifiers=["Programming Language :: Python :: 3"],
)
