# tests/data/media

Fixtures for `mm_test`. Solid red: a picture in eleven of the still formats `core/mm/media.cpp`
admits (PNG, APNG, JPEG, WebP, GIF, BMP, TIFF, QOI, PPM, JPEG-LS, AVIF) and a short clip in nine of
the container and codec pairs it admits. The test decodes each one and checks that it is red at its
size. They were made with FFmpeg 8.1 (libx264, libx265, libvpx, libsvtav1, libaom, libtheora,
libwebp):

```sh
I="-f lavfi -i color=c=red:s=32x32:d=1:r=1"
V="-f lavfi -i color=c=red:s=64x64:d=1:r=4"
ffmpeg $I -frames:v 1 red.png
ffmpeg $I -frames:v 2 -f apng red.apng
ffmpeg $I -frames:v 1 red.jpg
ffmpeg -f lavfi -i color=c=red:s=1788x282:d=1:r=1 -frames:v 1 red-1788x282.jpg
ffmpeg $I -frames:v 1 -c:v libwebp red.webp
ffmpeg $I -frames:v 1 red.gif
ffmpeg $I -frames:v 1 red.bmp
ffmpeg $I -frames:v 1 red.tiff
ffmpeg $I -frames:v 1 -c:v qoi red.qoi
ffmpeg $I -frames:v 1 -pix_fmt rgb24 red.ppm
ffmpeg $I -frames:v 1 -c:v jpegls red.jls
ffmpeg $V -frames:v 1 -c:v libaom-av1 -cpu-used 8 -f avif red.avif
ffmpeg $V -c:v libx264 -pix_fmt yuv420p red-h264.mp4
ffmpeg $V -c:v libx265 -pix_fmt yuv420p -tag:v hvc1 red-hevc.mp4
ffmpeg $V -c:v libsvtav1 -pix_fmt yuv420p red-av1.mp4
ffmpeg $V -c:v libvpx red-vp8.webm
ffmpeg $V -c:v libvpx-vp9 red-vp9.webm
ffmpeg $V -c:v libx264 -pix_fmt yuv420p red-h264.mkv
ffmpeg $V -c:v libx264 -pix_fmt yuv420p -f mpegts red-h264.ts
ffmpeg $V -c:v mpeg4 red-mpeg4.avi
ffmpeg $V -c:v libtheora red-theora.ogv
```

`red-1788x282.jpg` is 4:2:0 at a width the scaler's vectors do not divide: converting it to RGB,
the scaler writes 12 bytes past width x 3, and
`a_picture_of_an_awkward_width_converts_inside_its_buffer` checks that the output buffer holds
them.

The Theora clip repeats its first frame with empty packets, which is how Ogg Theora encodes a frame
that does not change.

`quad.png`, `quad.bmp`, `quad.jpg` (quality 95, 4:4:4), `quad.webp` (lossless) and `quad-alpha.png`
are written by Pillow (`python3 tests/mkimggolden.py`): 33 x 17 pixels, red top left, green top
right, blue bottom left, white bottom right, split at x = 16 and y = 8. A decoder that flips the
row order (BMP stores rows bottom up), mirrors a row or takes the wrong stride at an odd width puts
a wrong colour in some quadrant, which a solid picture cannot show.
`pictures_decode_the_right_way_up_and_the_right_way_round` checks them. `quad-alpha.png` carries
the same colours with alpha round(255 x / 32) in column x, for
`the_rgba_decode_keeps_alpha_and_the_rgb_decode_drops_it_uncomposited`.
