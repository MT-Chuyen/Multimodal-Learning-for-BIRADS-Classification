from train import trainer
from testing import tester
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,  choices=['cross_attn','fusion'],default="cross_attn", help="Model name")
    parser.add_argument("--epoch", type=int,default=100)
    parser.add_argument("--loss", type=str, default='cross_entropy', choices= ['cross_entropy', 'focal_loss', 'hierarchical', 'attention_entropy', 'weighted_multitask'])
    parser.add_argument('--self_meta', type=bool, default=False,  help='Use self-attention for metadata')
    parser.add_argument('--self_image', type=bool, default=False, help='Use self-attention for image')
    parser.add_argument('--cross_attn', type=bool, default=True,  help='Use cross-modality attention')
    parser.add_argument('--num_classes',type=int,choices = [3,5], default=3)

    parser.set_defaults(self_meta=False, self_image=False, cross_attn=True)

    args = parser.parse_args()
    trainer(args)
    tester(args)