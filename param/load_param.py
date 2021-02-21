import yaml
import pprint
from argparse import Namespace


def load_config(config_file):
    f = open(config_file, "r")
    cfg = yaml.load(f)
    return cfg


def pprint_cfg(cfg):
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(cfg)


def to_namespace_recursive(cfg: dict) -> Namespace:
    ns = Namespace()
    kv = vars(ns)
    for k, v in cfg.items():
        if isinstance(v, dict):
            sub_ns = to_namespace_recursive(v)
            kv[k] = sub_ns
        else:
            kv[k] = v
    return ns



if __name__ == "__main__":
    config_file = "/Users/francis/code/fastspeech2/param/config.yaml"
    cfg = load_config(config_file)
    ns = to_namespace_recursive(cfg)
    print(ns)