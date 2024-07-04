.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/LORE_sa.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/LORE_sa
    .. image:: https://readthedocs.org/projects/LORE_sa/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://LORE_sa.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/LORE_sa/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/LORE_sa
    .. image:: https://img.shields.io/pypi/v/LORE_sa.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/LORE_sa/
    .. image:: https://img.shields.io/conda/vn/conda-forge/LORE_sa.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/LORE_sa
    .. image:: https://pepy.tech/badge/LORE_sa/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/LORE_sa
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/LORE_sa

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

=======
LORE Stable and Actionable
=======

This is the official repository of the LORE (Local Rule-Based Explanation) algorithm.

LORE is a model-agnostic algorithm that learns to explain any machine learning model. It is a local explanation algorithm that generates explanations for individual predictions. LORE buils a synthetic neighborhood around the instance to be explained and learns a transparent decision tree model to approximate the behavior of the underlying model in that neighborhood. The tree is exploited to generate explanations in terms of logical rules.


Getting Started
===============
We suggest to install the library and its requirements into a dedicated environment.

.. code-block:: bash

    virtualenv -p python3 venv
    source venv/bin/activate
    pip install -e .


The last command will create a development installation of the library, so you can modify the code and see the changes immediately. We are still working to provide a package to install under Conda.

To use the library within your project just import the needed packages:

.. code-block:: python
   from src.lore_sa.dataset import TabularDataset

   # load the training data
   dataset = TabularDataset.from_csv('my_data.csv', class_name = "class")


Your first LORE explanation
===========================
Let's consider a simple example to explain the prediction of a model on a tabular dataset. We have a tabular dataset
containing observation of a Credit Risk use case. We will create an opaque model using a Random Forest classifier and
then we will use LORE to explain the prediction of the model on a specific instance.

Let's start by loading the dataset and creating the model:

.. code-block:: python
      import pandas as pd
      from sklearn.ensemble import RandomForestClassifier
      from sklearn.compose import ColumnTransformer
      from sklearn.preprocessing import StandardScaler, OrdinalEncoder
      from sklearn.pipeline import make_pipeline
      from sklearn.model_selection import train_test_split

      from lore_sa import sklearn_classifier_bbox

      df = pd.read_csv('data/credit_risk.csv')
      preprocessor = ColumnTransformer(
          transformers=[
              ('num', StandardScaler(), [0,8,9,10]),
              ('cat', OrdinalEncoder(), [1,2,3,4,5,6,7,11])
          ]
      )
      model = make_pipeline(preprocessor, RandomForestClassifier(n_estimators=100, random_state=42))

      X_train, X_test, y_train, y_test = train_test_split(df.loc[:, 'age':'native-country'].values, df['class'].values,
                                                  test_size=0.3, random_state=42, stratify=df['class'].values)
      model.fit(X_train, y_train)

      bbox = sklearn_classifier_bbox.sklearnBBox(model)


We wrap the model into a ``sklearnBBox`` object, which is a wrapper that allows LORE to interact with the model.
Now we can use LORE to explain the prediction of the model on a specific instance.
We need to provide LORE with information on the internal structure of the dataset. We will use the ``TabularDataset``
class to do so. ::

   from src.lore_sa.dataset import TabularDataset
   from lore_sa.generator import GeneticGenerator
   from src.lore_sa.encoder_decoder import ColumnTransformerEnc
   from src.lore_sa.lore import Lore
   from src.lore_sa.surrogate import DecisionTreeSurrogate

   dataset = TabularDataset.from_csv('resources/adult.csv', class_name='class')
   dataset.df.dropna(inplace=True)
   dataset.df.drop(['fnlwgt', 'education-num'], axis=1, inplace=True)
   dataset.update_descriptor()

   enc = ColumnTransformerEnc(dataset.descriptor)
   generator = GeneticGenerator(bbox, dataset, enc)
   surrogate = DecisionTreeSurrogate()

   tabularLore = Lore(bbox, dataset, encoder, generator, surrogate)


Now we have an instance of the ``Lore`` class that we can use to explain the prediction of the model on a specific instance.
Let's consider the first instance of the test set and explain the prediction of the model on this instance::

   instance = X_test[0]
   explanation = tabularLore.explain_instance(instance)
   print(explanation)


Issue tracking
==============

For any issue or bug, please open a new issue in the issue tracker available at: https://github.com/kdd-lab/LORE_sa/issues

Contributing
============

If you want to contribute to the library, please fork the repository and submit a pull request with the changes. The pull request will be reviewed by the maintainers and merged into the main branch if the changes are considered appropriate.


Documentation
=============

The documentation is based on Sphinx. Documentation of the code is created by simply writing docstrings using reStructuredText markup. Docstrings are comments placed within triple quotes (''' or """) immediately below module, class, function, or method definitions.

The creation of online documentation the features of Sphinx.
To build the documentation::

   cd docs
   make html



Once the documentation is built, the new folder ``docs/html`` must be committed and pushed to the repository and the documentation is then available here: https://kdd-lab.github.io/LORE_sa/html/index.html

To update the online documentation, as an instance when new modules or function are added to the LORE_sa library, it is necessary to delete the old folder ``docs/html``, build the documentation (see the snippet above)  and copy the greshly created ``docs/_build/html`` folder into ``docs/``. Then, after committing and pushing the folder ``docs/html``, the online documentation is updated to the last version.




.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.2.2. For details and usage
information on PyScaffold see https://pyscaffold.org/.
