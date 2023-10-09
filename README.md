# LORE_sa
Code for LORE (under refactoring)

## Documentation

The documentation is provides by Sphinx and the user have to build it to deploy it. 

```bash

cd docs
make html

```

Delete the old folder `docs/html` and copy the `docs/_build/html` folder into `docs/`.
Push the commit and wait for the action pipelin `pages build and deployment` is successfully completed.

Online documentation avalaible here: https://kdd-lab.github.io/LORE_sa/html/index.html

ATT. Not work on `gh-pages` branch