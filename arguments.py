from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def get_parser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--wsi",
                        type=str,
                        help="path to the wsi folder", 
                        required=True)
    parser.add_argument("--xml", 
                        type=str, 
                        help="path to the xml folder", 
                        default='.')
    parser.add_argument("--table_data", 
                        type=str,
                        help="path to the table_data.csv", 
                        required=True)
    parser.add_argument("--n_sample", 
                        type=int,
                        help="number of patches sampled per wsi", 
                        default=1)
    parser.add_argument("--color_aug", 
                        type=int,
                        help="If there is color augmentation or not",
                        default=0)
    parser.add_argument("--epochs", 
                        type=int,
                        help="number of epochs to train",
                        default=10)
    parser.add_argument("--batch_size",
                        type=int, 
                        help="Batch size", 
                        default=10)
    parser.add_argument("--patience", 
                        type=int,
                        help="Number of epochs without improvements before stopping",
                        default=5)
    parser.add_argument("--resolution", 
                        type=int,
                        help="resolution at which to sample",
                        default=0)
    parser.add_argument('--model_name',
                        type=str,
                        help='Name of the network to use: resnet18 | resnet50',
                        default='resnet18')
    parser.add_argument('--pretrained',
                        type=int,
                        help='If >=1 loads the weights trained on imagenet',
                        default=0)
    parser.add_argument('--frozen',
                        type=int,
                        help='If >= 1, freeze the network except fc layer',
                        default=0)
    
    return parser