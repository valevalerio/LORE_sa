# LORE_sa
Code for LORE (under refactoring)

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


