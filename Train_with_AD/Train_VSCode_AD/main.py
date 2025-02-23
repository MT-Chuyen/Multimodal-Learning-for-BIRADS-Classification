from train import trainer
from testing import tester
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", 
        type=str,  
        choices=[
            'efficientnetb0', 'densenet201', 'resnet152', 'mobilenetv3'
        ], 
        default="normal", 
        help="Model name"
    )
    parser.add_argument("--epoch", type=int, default=100, help="Number of epochs")
    parser.add_argument("--loss", type=str, default='cross_entropy', choices= ['cross_entropy', 'focal_loss'])
    parser.add_argument("--selfsa", type=bool, choices= [True, False], default=False)
    args = parser.parse_args()

    print(f"Selected Model: {args.model}")
    print(f"Number of Epochs: {args.epoch}")

    trainer(args)
    tester(args)
