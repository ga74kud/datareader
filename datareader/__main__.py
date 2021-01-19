import argparse
import definitions
from __init__ import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_x', '-wix', type=int, help='windowsize in x-direction for savgol_filter', default=101,
                        required=False)
    parser.add_argument('--window_y', '-wiy', type=int, help='windowsize in y-direction for savgol_filter', default=101,
                        required=False)
    parser.add_argument('--poly_x', '-pox', type=int, help='polygon order in x-direction for savgol_filter', default=2,
                        required=False)
    parser.add_argument('--poly_y', '-poy', type=int, help='polygon order in y-direction for savgol_filter', default=2,
                        required=False)
    args = parser.parse_args()
    params = vars(args)
    params['PROJECT_ROOT'] = definitions.get_project_root_pkg()

if __name__ == '__main__':
    main()