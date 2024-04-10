# LORE_sa
This is the official repository of the LORE (Local Rule-Based Explanation) algorithm. 

## Getting started

We suggest to install the library and its requirements into a dedicated environment.
```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt 
```

To use the library within your project just import the needed packages:
```python
from lore_sa.dataset import TabularDataset

# load the training data
dataset = TabularDataset.from_csv('my_data.csv', class_name = "class")

```

## Issue tracking
For any issue or bug, please open a new issue in the issue tracker available at: https://github.com/kdd-lab/LORE_sa/issues

## Contributing
If you want to contribute to the library, please fork the repository and submit a pull request with the changes. The pull request will be reviewed by the maintainers and merged into the main branch if the changes are considered appropriate.


## Documentation

The documentation is based on Sphinx. Documentation of the code is created by simply writing docstrings using reStructuredText markup. Docstrings are comments placed within triple quotes (''' or """) immediately below module, class, function, or method definitions.

The creation of online documentation the features of Sphinx. 
To build the documentation:  

```bash

cd docs
make html

```
Once the documentation is built, the new folder `docs/html` must be committed and pushed to the repository and the documentation is then available here: https://kdd-lab.github.io/LORE_sa/html/index.html

To update the online documentation, as an instance when new modules or function are added to the LORE_sa library, it is necessary to delete the old folder `docs/html`, build the documentation (see the snippet above)  and copy the greshly created `docs/_build/html` folder into `docs/`. Then, after committing and pushing the folder `docs/html`, the online documentation is updated to the last version.


