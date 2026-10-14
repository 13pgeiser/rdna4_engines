/* imaging.h -- the picture arithmetic an image generator's reference pipeline does in PIL, done the
 * way PIL does it, so a picture reaches the model as the reference's would.
 *
 * THE RESIZE IS PIL's Image.resize WITH LANCZOS, bit for bit: per-axis filter coefficients
 * computed in double (support 3 times the downscale, normalised to sum one), turned into 22-bit
 * fixed point, a horizontal pass then a vertical one over 8-bit values, each output rounded and
 * clipped -- and an RGBA picture premultiplied by its alpha first and divided back after, which is
 * what PIL's resize does to RGBA. Diffusers' VaeImageProcessor resizes a picture exactly this way.
 *
 * THE COMPOSITE is Image.paste(picture, mask=alpha) onto white: each channel
 * (c * a + 255 * (255 - a)) / 255 in PIL's rounding -- how the reference flattens a transparent
 * picture for a text encoder that reads RGB. */
#pragma once
#include "mm/media.h"

namespace rad {
namespace mm {

/* `src` resized to w x h. */
void resize_lanczos_rgba(const FrameRGBA& src, int32_t w, int32_t h, FrameRGBA* out);
/* `src` over an opaque white ground. */
void composite_white(const FrameRGBA& src, Frame* out);

}  /* namespace mm */
}  /* namespace rad */
