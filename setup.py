from setuptools import setup, find_packages

setup(
    name='events_synergy',  # Replace 'your_package_name' with the actual name of your package
    version='0.1.0',  # Set the version of your package
    description='Events Synergetic Training',  # Provide a short description
    long_description=open('README.md').read(),  # This will read your README file as the long description
    long_description_content_type='text/markdown',  # Set the content type to markdown
    packages=find_packages(),  # Automatically find and include all sub-packages
    include_package_data=True,  # Include other files specified in MANIFEST.in
    install_requires=[
        'jinja2',
        'pytest',
        'transformers',
        'datasets',
        'evaluate',
        'tqdm',
        'typer',
        'rouge_score',
        'absl-py',
        'accelerate'
    ],
    dependency_links=[
        'git+https://github.com/ns-moosavi/coval#egg=coval'
    ],
    python_requires='>=3.7',  # Specify the python version requirements
)

