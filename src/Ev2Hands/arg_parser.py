import argparse
import os

def demo():
    parser = argparse.ArgumentParser(description='Demo for Ev2Hands')

    parser.add_argument('--batch_size', dest='batch_size', required=False,
                        help='Set the batch_size (default: 128)', default='32')

    parser.add_argument('--checkpoint_path', dest='checkpoint', required=False,
                        help='path of checkpoint_path', default='./savedmodels/best_model_state_dict.pth')

    args = parser.parse_args()
        
    os.environ['CHECKPOINT_PATH'] = args.checkpoint
    os.environ['BATCH_SIZE'] = args.batch_size

    return args


def evaluate():
    parser = argparse.ArgumentParser(description='Evaluation of Ev2Hands')
    

    parser.add_argument('--batch_size', dest='batch_size', required=False,
                        help='Set the batch_size (default: 128)', default='128')

    parser.add_argument('--checkpoint_path', dest='checkpoint', required=False,
                        help='path of checkpoint',
                        default='./savedmodels/best_model_state_dict.pth')
 
    args = parser.parse_args()

    os.environ['CHECKPOINT_PATH'] = args.checkpoint
    os.environ['BATCH_SIZE'] = args.batch_size

    return args


def train():
    parser = argparse.ArgumentParser(description='Trainer of Ev2Hands')
    
    parser.add_argument('--batch_size', dest='batch_size', required=False,
                        help='Set the batch_size (default: 8)', default='8')

    parser.add_argument('--checkpoint_path', dest='checkpoint', required=False,
                        help='path of checkpoint', default='')
 
    args = parser.parse_args()

    os.environ['CHECKPOINT_PATH'] = args.checkpoint
    os.environ['BATCH_SIZE'] = args.batch_size

    return args


def finetune():
    parser = argparse.ArgumentParser(description='FineTuner of Ev2Hands for real data')
    
    parser.add_argument('--batch_size', dest='batch_size', required=False,
                        help='Set the batch_size (default: 8)', default='8')

    parser.add_argument('--checkpoint_path', dest='checkpoint', required=False,
                        help='path of checkpoint', 
                        default='./savedmodels/best_model_state_dict.pth')
 
    args = parser.parse_args()

    os.environ['CHECKPOINT_PATH'] = args.checkpoint
    os.environ['BATCH_SIZE'] = args.batch_size

    return args