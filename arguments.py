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
                        required=True)
    parser.add_argument("--table_data", 
                        type=str,
                        help="path to the table_data.csv", 
                        required=True)
    parser.add_argument("--n_sample", 
                        type=int,
                        help="number of patches sampled per wsi", 
                        default=1)
    parser.add_argument("--color_aug", 
                        type=bool,
                        help="If there is color augmentation or not",
                        default=False)
    parser.add_argument('--model', 
                        type=str,
                        help='name of the classifier model', 
                        default='resnet18')
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
    parser.add_argument("--seed", 
                        type=int,
                        help="Random seed", 
                        default=42)
    return parser
