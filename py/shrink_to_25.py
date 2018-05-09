# -*- coding: utf-8 -*-
"""
Created on 2018-05-09 11:15:15
@Author: ZHAO Lingfeng
@Version : 0.0.1
"""
import os

from PIL import Image


def main():
    while 1:
        file = input("input image path or drag image here:\n")
        try:
            image = Image.open(file)
            x, y = image.size
            image = image.resize((x // 2, y // 2))
            image.save(file)
            print(os.path.basename(file), " Done!")

        except Exception as e:
            print("wrong image file: ", e)
            continue


if __name__ == "__main__":
    main()
