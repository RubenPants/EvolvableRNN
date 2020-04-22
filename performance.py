import argparse
from timeit import timeit


def test_intersection():
    print("--> Running the intersection test <--")
    
    # Do measures
    time_cy = timeit('intersection_test_cy.main()',
                     setup='from tests.cy import intersection_test_cy',
                     number=1000)
    time_py = timeit('intersection_test.main()',
                     setup='from tests import intersection_test',
                     number=1000)
    
    # Print results
    print("Time Python:", time_py)
    print("Time Cython:", time_cy)
    print(f"Cython is {round(time_py / time_cy, 2)} times faster on the intersection test")


def test_drive():
    print("--> Running the drive test <--")
    
    # Do measures
    time_cy = timeit('drive_test_cy.main()',
                     setup='from tests.cy import drive_test_cy',
                     number=10)
    time_py = timeit('drive_test.main()',
                     setup='from tests import drive_test',
                     number=10)
    
    # Print results
    print("Time Python:", time_py)
    print("Time Cython:", time_cy)
    print("Cython is {} times faster on the drive test".format(round(time_py / time_cy, 2)))


def test_sensors():
    print("--> Running the sensor test <--")
    
    # Do measures
    time_cy = timeit('sensors_test_cy.main()',
                     setup='from tests.cy import sensors_test_cy',
                     number=1000)
    time_py = timeit('sensors_test.main()',
                     setup='from tests import sensors_test',
                     number=1000)
    
    # Print results
    print("Time Python:", time_py)
    print("Time Cython:", time_cy)
    print("Cython is {} times faster on the sensor test".format(round(time_py / time_cy, 2)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--eval', type=str, default='all')
    args = parser.parse_args()
    
    x = args.eval
    if x == 'all' or x == 'intersection':
        test_intersection()
    if x == 'all' or x == 'drive':
        test_drive()
    if x == 'all' or x == 'sensors':
        test_sensors()
