import os


def autoRun():
    for radius in range(1, 8):
        os.system(f"./build/cuda_guided_filter {radius} 0.3 1000 data/adobe_image_4.jpg data/adobe_gt_4.jpg")


if __name__ == "__main__":
    autoRun()

