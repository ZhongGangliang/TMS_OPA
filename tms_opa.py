import argparse
import os
from tms_evaluation import cal_tms_evalution
from analysis_config import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analysis TMS evaluation')
    parser.add_argument("-o", "--output", type=str,
                        help="Output folder")
    args = vars(parser.parse_args())

    
    output_path = dataPath + 'analysis_path/' + args['output'] + '/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    cal_tms_evalution(output_path, subjectnum_result)