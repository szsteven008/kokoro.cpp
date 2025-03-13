import argparse
import torch
import json

def export_voice(source, target):
    pack = torch.load(source, weights_only=True).squeeze(1)
    print(f'pack: {pack.shape}')
    print(f'pack: {pack}')

    with open(target, 'w') as f:
        json.dump(pack.numpy().tolist(), f)

    print(f'export {source} to {target} ok!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Export kokoro Voice Model to bin", add_help=True)
    parser.add_argument("--input", "-i", type=str, required=True, help="path to source(pt) file")
    parser.add_argument("--output", "-o", type=str, required=True, help="path to target(json) file")

    args = parser.parse_args()

    # cfg
    source = args.input  # change the path of the model config file
    target = args.output  # change the path of the model

    export_voice(source, target)