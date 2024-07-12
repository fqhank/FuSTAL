import os
import argparse

def run_full(args):
    os.system('python CrossVideo-Generator/train_generator.py')
    os.system('bash Student-Training/tools/thumos_i3d.sh 0')

if __name__ == 'main':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',type=str,default='all')
    # parser.add_argument('--store_generator_at',type=str,default='Student_Trainer/')
    args = parser.parse_args()
    
    if parser.mode == 'all':
        run_full(args)