import argparse
import logging
import sys
from datetime import datetime
import pandas as pd
from sklearn.metrics import classification_report
from models import DfModel as DM
from utils import log_args, LoggerWritter
from als import ErrorAnalysis

# logging
handler = logging.StreamHandler(sys.stdout)
log = logging.getLogger('analysis')
log.addHandler(handler)
log.setLevel(logging.INFO)
sys.stderr = LoggerWritter(log.warning)


def go_analysis(args):
    log.info('Begin analysis:')
    als = ErrorAnalysis(args)
    log.info('Save to json file:')
    als.save_to_json()


def add_args(args):
    # loss_type
    if args.loss_type is None:
        if args.task_type == 'real':
            args.loss_type = ['r2_score', 'rmse_loss', 'mse_loss']
        else:
            args.loss_type = ['cross_entropy']

    # process loss_kwargs
    if args.loss_kwargs is None:
        args.loss_kwargs = [{}] * len(args.loss_type)
    else:
        tmp_kwargs = []
        for kwargs in args.loss_kwargs:
            tmp_kwargs.append(eval(kwargs))
        args.loss_kwargs = tmp_kwargs

    # process reg_heatmap_kwargs
    if args.reg_heatmap_kwargs is None:
        args.reg_heatmap_kwargs = {}
    else:
        args.reg_heatmap_kwargs = eval(args.reg_heatmap_kwargs)

    return args


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # read csv file
    parser.add_argument('--csv_input_path', type=str, default=None)
    parser.add_argument('--pred_col', type=str, default='y_pred')
    parser.add_argument('--truth_col', type=str, default='y_truth')

    # loss
    parser.add_argument('--loss_type', nargs='*', type=str, default=None,
                        choices=['r2_score', 'rmse_loss', 'mse_loss', 'cross_entropy'])
    parser.add_argument('--loss_kwargs', nargs='*', type=str, default=None,
                        help='format should be like: "{"key1":"value1"} {"key1":"value1","key2":"value2"} {}"')

    # task_type
    parser.add_argument('--task_type', type=str, default='categorical',
                        choices=['binary', 'categorical', 'real'])

    # output dir
    parser.add_argument('--json_output_path', type=str, default=None)
    parser.add_argument('--img_save_dir', type=str, default=None)

    # regression heatmap kwargs
    parser.add_argument('--reg_heatmap_kwargs',
                        nargs='*', type=str, default=None, help='format should be like: "{"thres":20,"n_blocks":5}"')

    args = parser.parse_args()

    log_args(args, log)
    args = add_args(args)
    go_analysis(args)
