import unittest

import pathlib

from helpers.data_augment import generate_img_filelist_txt


class DataAugmentTestCase(unittest.TestCase):
    def test_generate_filelist_for_train(self):
        root = pathlib.Path('../data') / 'aug_widerface' / 'AUG_WIDER_train' / 'images'
        destination = pathlib.Path(
            '../data') / 'aug_widerface' / 'AUG_wider_face_split' / 'aug_wider_face_train_filelist.txt'
        generate_img_filelist_txt(root, destination)
        self.assertTrue(destination.exists())

    def test_generate_filelist_for_val(self):
        root = pathlib.Path('../data') / 'aug_widerface' / 'AUG_WIDER_val' / 'images'
        destination = pathlib.Path(
            '../data') / 'aug_widerface' / 'AUG_wider_face_split' / 'aug_wider_face_val_filelist.txt'
        generate_img_filelist_txt(root, destination)
        self.assertTrue(destination.exists())




if __name__ == '__main__':
    unittest.main()
