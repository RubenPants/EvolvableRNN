"""
process_killer.py

Kill sleeping processes.
"""
import argparse
import os
import sys

import psutil


def write_all_processes():
    temp = sys.stdout  # store original stdout object for later
    sys.stdout = open('out.txt', 'w')  # redirect all prints to this log file
    psutil.test()
    sys.stdout.close()
    sys.stdout = temp  # restore print commands to interactive prompt


def parse_process_file(file_name):
    with open('out.txt', 'r') as f:
        lines = f.read().split('\n')
        
        # Fetch all the sleeping processes that contain the file-name
        sleeping_processes = []
        for line in lines:
            if (' sleep ' in line) and (file_name in line): sleeping_processes.append(line)
        
        # Parse out the PIDs and terminate the processes
        for sp in sleeping_processes:
            sp = [s for s in sp.split(' ') if len(s) > 0]
            pid = int(sp[1])  # Second item contains the PID
            p = psutil.Process(pid)  # Get control of the sleeping process
            p.terminate()  # Terminate the process
    
    # Close the process-file
    os.remove('out.txt')


def main(file_name):
    """Run the process-killer."""
    write_all_processes()
    parse_process_file(file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--file_name', type=str, default='main.py')
    args = parser.parse_args()
    
    # Run the process
    main(args.file_name)
