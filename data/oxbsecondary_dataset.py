"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os.path
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset


class OxbsecondaryDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)

        parser.set_defaults(preprocess_mode='fixed')
        parser.set_defaults(load_size=512)
        parser.set_defaults(crop_size=512)
        parser.set_defaults(display_winsize=512)
        parser.set_defaults(label_nc=35)
        parser.set_defaults(aspect_ratio=2.0)
        parser.set_defaults(batchSize=16)
        opt, _ = parser.parse_known_args()
        if hasattr(opt, 'num_upsampling_layers'):
            parser.set_defaults(num_upsampling_layers='more')
        return parser

    def get_paths(self, opt):
        root = opt.dataroot_secondary
        phase = 'val' if opt.phase == 'test' else 'train'

        label_dir = os.path.join(root)
        label_paths_all = make_dataset(label_dir, recursive=True)
        label_paths = [p for p in label_paths_all if p.endswith('_labelmapcity.png')]
        image_paths = [p.replace("_labelmapcity.png", "_rgb.png") for p in label_paths]

        if not opt.no_instance:
            instance_paths = [p.replace("_labelmapcity.png", "_instancemap.png") for p in label_paths]
        else:
            instance_paths = []

        return label_paths, image_paths, instance_paths

    def paths_match(self, path1, path2):
        return True
