import unittest
from lore_sa.rule.rule import Expression
import operator



class RuleTest(unittest.TestCase):
    pass


class ExpressionTest(unittest.TestCase):

    def testExpression(self):
        exp = Expression("test", operator.gt,10)
        self.assertEqual(str(exp),"test > 10")
    

if __name__ == '__main__':
    unittest.main()
