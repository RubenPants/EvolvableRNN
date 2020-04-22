"""
run_tests.py

Run all the tests.
"""
import sys
import unittest

if __name__ == '__main__':
    # TODO: Uncomment and add to suite if you want to test with PyTorch (test validity of RNN cells)
    # suite_torch = unittest.TestLoader().discover('.', pattern="*_test_torch.py")
    # suite_torch_cy = unittest.TestLoader().discover('.', pattern="*_test_torch_cy.py")
    if 'linux' in sys.platform:
        suite_py = unittest.TestLoader().discover('.', pattern="*_test.py")
        suite_cy = unittest.TestLoader().discover('.', pattern="*_test_cy.py")
        suite = unittest.TestSuite([suite_py, suite_cy])
    else:
        suite = unittest.TestLoader().discover('.', pattern="*_test.py")
    unittest.TextTestRunner(verbosity=2).run(suite)
