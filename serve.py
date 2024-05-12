import argparse
from lilush_llm_backend import Serve

parser = argparse.ArgumentParser()
parser.add_argument("--ip", type=str, default="127.0.0.1", required=False)
parser.add_argument("-p", "--port", type=int, default=8013, required=False)
args = parser.parse_args()

Serve(args.ip, args.port)
