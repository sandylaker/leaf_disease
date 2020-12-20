import torch
import os.path as osp
from argparse import ArgumentParser
from collections import OrderedDict


def remove_prefix(text, prefix):
    return text[len(prefix):] if text.startswith(prefix) else text


def clear_checkpoint(ckpt_file):
    ckpt = torch.load(ckpt_file, map_location='cpu')
    assert 'state_dict' in ckpt

    state_dict = ckpt['state_dict']
    state_dict_cleared = OrderedDict([(remove_prefix(k, 'model.'), v) for
                                      k, v in state_dict.items()])

    save_path = osp.splitext(ckpt_file)[0] + '.pth'
    torch.save(state_dict_cleared, save_path)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('file', type=str, help='check point file to be processed')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    clear_checkpoint(args.file)


if __name__ == '__main__':
    main()