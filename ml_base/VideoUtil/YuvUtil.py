from common_lib_import_and_set import *


def yuv2RGB(y, u, v):
    r = 1.164 * (y - 16) + 1.596 * (v - 128)
    r = max(min(r, 255), 0)
    g = 1.164 * (y - 16) - 0.813 * (v - 128) - 0.391 * (u - 128)
    g = max(min(g, 255), 0)

    b = 1.164 * (y - 16) + 2.018 * (u - 128)
    b = max(min(b, 255), 0)

    return int(r), int(g), int(b)


if __name__ == "__main__":

    print(yuv2RGB(81, 90, 239))
    print(yuv2RGB(144, 51, 33))
    print(yuv2RGB(144, 51, 33))
