import os
import re

from srcs.vision import show_cont_vision,show_fill_vision
from srcs.calculator import run_eva,run_eva_dic,run_eva_single
from args import get_parser


if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

    parser = get_parser()
    args = parser.parse_args()
    if args.function:
        func_list = re.split(r',',args.function)
        for fun in func_list:
            print(f'Running:{fun}')
            locals()[fun](**vars(args))
            print('\n')

        print('Complete.')
