import unittest
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from lore_sa.surrogate import DecTree

class SurrogateTest(unittest.TestCase):

    def test_init(self):

        iris = load_iris()
        dt = DecTree().train(iris.data, iris.target)
        self.assertIsInstance(dt, DecisionTreeClassifier)


if __name__ == '__main__':
    unittest.main()
