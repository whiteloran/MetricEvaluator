import argparse

def get_parser():

    parser = argparse.ArgumentParser(description='Evaluator')
    
    # function
    parser.add_argument('-dataset_name',default = None, type=str)
    parser.add_argument('-func', dest='function', default = None, type=str)
    parser.add_argument('-classes', default = 1, type=int)

    # image
    parser.add_argument('-img_h', default=80, type=int)
    parser.add_argument('-img_w', default=80, type=int)

    # evaluate
    parser.add_argument('-img_path', dest='img_path', default = None, type=str)
    parser.add_argument('-gt_path', dest='gt_path', default = None, type=str)
    parser.add_argument('-pred_path',dest='pred_path', default= None, type=str)

    # vision
    parser.add_argument('-save_path',dest='save_path', default=None, type=str)

    # initialize
    parser.add_argument('-log_term', default = True, type=bool)

    return parser


if __name__ =="__main__":

    parser = get_parser()
    args_dict = parser.parse_args()