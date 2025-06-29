from operator import concat

import yaml


class ConfigModelStructure:
    def __init__(self, config_path: str = 'yolov5m.yaml'):
        self.config_path = config_path
        self.backbone = None
        self.head = None
        self.anchors = None
        self.load_config()

    def load_config(self):
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
            self.backbone = config.get('backbone', {})
            self.head = config.get('head', {})
            self.anchors = config.get('anchors', [])


def main():
    config = ConfigModelStructure('../model/yolov5m.yaml')
    layer = concat(config.backbone, config.head)
    print(layer[24])


if __name__ == "__main__":
    main()
