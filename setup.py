import ast
from ast import NodeVisitor
import os
from setuptools import setup, find_packages


class VersionExtractor(NodeVisitor):
    def __init__(self):
        super().__init__()
        self.version = None

    def visit_Assign(self, node):
        if hasattr(node.targets[0], "id") and node.targets[0].id == "__version__":
            self.version = node.value.s


def parse_version():

    with open(os.path.join("dgc_gnn", "__init__.py"), "r") as f:
        content = f.read()

    tree = ast.parse(content)
    visitor = VersionExtractor()
    visitor.visit(tree)
    return visitor.version


def long_description():
    with open("README.md", "r") as f:
        return f.read()


setup(
    name="dgc_gnn",
    version=parse_version(),
    license="MIT",
    long_description=long_description(),
    long_description_content_type="text/markdown",
    install_requires=[
        "bpnpnet @ git+https://github.com/SergioRAgostinho/bpnpnet.git@package#egg=bpnpnet",
        "numpy",
        "opencv-python",
        "Pillow",
        "pytorch-lightning",
        "scipy",
        "tqdm",
    ],
    extras_require=dict(
        full=[
            "wget",
            "colmap @ git+https://github.com/SergioRAgostinho/colmap.git@python-packaging#egg=colmap",
        ]
    ),
    author="Shuzhe Wang, Juho Kannala, Daniel Barath",
    author_email="shuzhe.wang@aalto.fi",
    packages=find_packages(),
    python_requires=">=3.7",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
    ],
    acknowledgement="GoMatch: A Geometric-Only Matcher for Visual Localization"
)
