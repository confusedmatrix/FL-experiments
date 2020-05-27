import os
from setuptools import setup

def get_requirements(req_file):
    """Read requirements file and return packages and git repos separately"""
    requirements = []
    dependency_links = []
    lines = open(req_file).read().split("\n")
    for line in lines:
        if line.startswith("git+"):
            dependency_links.append(line)
        else:
            requirements.append(line)
    return requirements, dependency_links

core_reqs, core_dependency_links = get_requirements("requirements.txt")

setup(
    name='fl_experiments',
    version='0.1.0',
    description='A framework for running federated learning experiments',
    url='#',
    author='confusedmatrix',
    author_email='confusedmatrix@gmail.com',
    license='MIT',
    install_requires=core_reqs,
    dependency_links=core_dependency_links,
    packages=['fl_experiments'],
    zip_safe=False
)