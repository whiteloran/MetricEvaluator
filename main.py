import os
import re
import sys

from srcs.vision import show_cont_vision,show_fill_vision
from srcs.calculator import run_eva,run_eva_dic,run_eva_single
from args import get_parser


if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

    parser = get_parser()
    args = parser.parse_args()
    if args.function:
        
        if args.log_term:
            print (f"Eval logs will be saved to: ./logs/{args.dataset_name}_eval.log")
            sys.stdout = open(os.path.join(f'./logs/{args.dataset_name}_eval.log'), 'w')
            sys.stderr = open(os.path.join(f'./logs/{args.dataset_name}_error.log'), 'w')
        
        func_list = re.split(r',',args.function)
        for fun in func_list:
            print(f'Running:{fun}')
            locals()[fun](**vars(args))
            print('\n')

        print('Complete.')
