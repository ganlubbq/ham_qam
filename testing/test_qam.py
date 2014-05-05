#!/usr/bin/env python
# the above line should allow you to execute this from the command line
# ex just run: ./test_qam.py

# Add imports here
# Let's put all of the qam modulation, demodulation, etc in qam.py so we can
# easily import it
from qam import *

def main():
    # Example of how to add command line arguments
    #parser = argparse.ArgumentParser(description='Run lab 3 functions.')
    #parser.add_argument('function', choices=['ptt', 'response', 'morse', 'all'], help='pick a function to run')
    #parser.add_argument('plen', type=float, default=150.0, help='ptt length')
    #parser.add_argument('text', type=str, default='KK6KKT', help='text to send in morse')
    #parser.add_argument('-t', '--transmit', action='store_true', default=False, help='print additional output')
    #parser.add_argument('-v', '--verbose', action='store_true', default=False, help='print additional output')
    #parser.add_argument('-p', '--plot', action='store_true', default=False, help='show plots')
    #args = parser.parse_args()
    #args.function
    #args.transmit

    # Generate test data to send
    prefix = np.array([[0],[2],[10],[8]])
    bits= np.array([[1],[1],[1],[1],[1],[0],[2],[10],[8],[6],[11] , [0], [4], [2],[1], [2], [5], [6], [8], [6], [10], [3], [14],[0],[10]])

    # Modulate test data

    # Demodulate test data

    # Decode test data

    # Uncomment to show all plots at the end
    #show()


# This will run when you execute the program.
if __name__ == "__main__":
    main()
