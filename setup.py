from setuptools import setup, find_packages

setup(
    name='py_NPClab-Package',
    version='0.0.1',
    author='Steve Didienne',
    description='Analysis data from comportment and electrophysiology',
    packages=find_packages(),
    license='Free',
    url = 'https://github.com/NPC-lab-python/py_NPC-Lab_Packages',
    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 1 - Planning",
        "License :: OSI Approved",
        "Natural Language :: French",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Topic :: Communications",
    ],
    install_requires=['requirements']
)

