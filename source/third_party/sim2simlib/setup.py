"""Installation script for the 'trackerLab' Python package."""

from setuptools import setup

# Package metadata (hardcoded)
PACKAGE_NAME = "sim2simlib"
VERSION = "0.1.0"
AUTHOR = "renforce_dynamics"
MAINTAINER = "renforce_dynamics"
MAINTAINER_EMAIL = "renforce_dynamics"
DESCRIPTION = "sim2sim utils"
REPOSITORY = ""
KEYWORDS = ["extension", "isaacLab"]

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "numpy",
    "mujoco",
    "torch",
    "rich"
]

# Setup the package installation
setup(
    # Package name and metadata
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    keywords=KEYWORDS,
    url=REPOSITORY,
    license="BSD-3-Clause",
    install_requires=INSTALL_REQUIRES,
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
    ],
    zip_safe=False,
    packages=[PACKAGE_NAME],  # Package directory
)
