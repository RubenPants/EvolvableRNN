"""
difference.py

Visualize the difference between the python and the cython file.
"""
import difflib
import os

# name, path to python file, path to cython file
files = [
    ('multi environment',
     'environment/env_multi.py',
     'environment/cy/env_multi_cy.pyx'),
    ('game',
     'environment/game.py',
     'environment/cy/game_cy.pyx'),
    ('robots',
     'environment/robot.py',
     'environment/cy/robot_cy.pyx'),
    ('sensors',
     'environment/sensors.py',
     'environment/cy/sensors_cy.pyx'),
    ('Simple RNN',
     'population/utils/rnn_cell_util/simple_rnn.py',
     'population/utils/rnn_cell_util/cy/simple_rnn_cy.pyx'),
    ('GRU',
     'population/utils/rnn_cell_util/berkeley_gru.py',
     'population/utils/rnn_cell_util/cy/berkeley_gru_cy.pyx'),
    ('GRU-NR',
     'population/utils/rnn_cell_util/berkeley_gru_no_reset.py',
     'population/utils/rnn_cell_util/cy/berkeley_gru_no_reset_cy.pyx'),
    ('GRU-NU',
     'population/utils/rnn_cell_util/berkeley_gru_no_update.py',
     'population/utils/rnn_cell_util/cy/berkeley_gru_no_update_cy.pyx'),
    ('LSTM',
     'population/utils/rnn_cell_util/lstm.py',
     'population/utils/rnn_cell_util/cy/lstm_cy.pyx'),
    ('line2d',
     'utils/line2d.py',
     'utils/cy/line2d_cy.pyx'),
    ('vec2d',
     'utils/vec2d.py',
     'utils/cy/vec2d_cy.pyx'),
    ('activation functions',
     'population/utils/network_util/activations.py',
     'population/utils/network_util/cy/activations_cy.pyx'),
    ('feedforward net',
     'population/utils/network_util/feed_forward_net.py',
     'population/utils/network_util/cy/feed_forward_net_cy.pyx'),
    ('test drive',
     'tests/drive_test.py',
     'tests/cy/drive_test_cy.py'),
    ('test line2d',
     'tests/line_2d_test.py',
     'tests/cy/line_2d_test_cy.py'),
    ('test sensors',
     'tests/sensors_test.py',
     'tests/cy/sensors_test_cy.py'),
]


def match(python_file, cython_file):
    """
    Match the cython-file to the (original) python file.
    
    :param python_file: String representing the python-file
    :param cython_file: String representing the cython-file
    :return: Difference-lists
    """
    # Get git-wise diff file
    diff = difflib.unified_diff(python_file, cython_file, fromfile='py', tofile='cy', lineterm='')
    lines = [l for l in diff][2:]
    
    # Python-code (minus)
    py = []
    concat = False
    for l in lines:
        if not concat and l[0] in ['+', '-']:
            py.append("")
            concat = True
        elif l[0] not in ['+', '-']:
            concat = False
        
        # Add if necessary
        if l[0] == '-': py[-1] += f'\n{l[1:]}' if len(py[-1]) > 0 else f'{l[1:]}'
    
    # Cython-code (plus)
    cy = []
    concat = False
    for l in lines:
        if not concat and l[0] in ['+', '-']:
            cy.append("")
            concat = True
        elif l[0] not in ['+', '-']:
            concat = False
        
        # Add if necessary
        if l[0] == '+': cy[-1] += f'\n{l[1:]}' if len(cy[-1]) > 0 else f'{l[1:]}'
    
    # Both lists must be equally long
    assert len(py) == len(cy)
    
    # Remove empty segments
    to_remove = []
    for i_block in range(len(py)):
        if (py[i_block].replace(" ", "") == "") and (cy[i_block].replace(" ", "") == ""): to_remove.append(i_block)
    for rm in reversed(to_remove):
        del py[rm]
        del cy[rm]
    return py, cy


def pretty_print(py, cy):
    """Pretty print the two lists."""
    # Enroll the diff-blocks
    py_unrolled = [line.split("\n") for line in py]
    cy_unrolled = [line.split("\n") for line in cy]
    
    # Define the maximum length of a single line for both the py and cy segments
    max_py = max({len(line) for block in py_unrolled for line in block})
    max_cy = max({len(line) for block in cy_unrolled for line in block})
    
    # Enlarge the blocks such that they contain an equal amount of lines
    for i_block in range(len(py_unrolled)):
        while len(py_unrolled[i_block]) > len(cy_unrolled[i_block]):
            cy_unrolled[i_block].append("")
        while len(py_unrolled[i_block]) < len(cy_unrolled[i_block]):
            py_unrolled[i_block].append("")
        assert len(py_unrolled[i_block]) == len(cy_unrolled[i_block])
    
    # Print out the differences
    print(f"{'PYTHON':^{max_py}} | {'CYTHON':^{max_cy}}")
    print("-" * (max_py + 3 + max_cy))
    for i_block in range(len(py_unrolled)):
        for i_line in range(len(py_unrolled[i_block])):
            print(f"{py_unrolled[i_block][i_line]:{max_py}} | {cy_unrolled[i_block][i_line]:{max_cy}}")
        print("-" * (max_py + 3 + max_cy))


if __name__ == '__main__':
    os.chdir("..")
    for name, f_py, f_cy in files:
        print(f"\n\n\n==> ANALYZING: {name}\n")
        # Load in the files as a list, split on the new-line symbol
        with open(f_py, 'r') as f:
            contents_py = f.read().split('\n')
        with open(f_cy, 'r') as f:
            contents_cy = f.read().split('\n')
        
        # Match the two files with each other
        diff_py, diff_cy = match(contents_py, contents_cy)
        
        # Pretty print the difference of the two files
        pretty_print(diff_py, diff_cy)
