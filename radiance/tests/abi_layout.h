/* abi_layout.h -- what abi_layout_test's C translation unit (abi_c_test.c) hands its C++ one: the
 * layout of every struct abi/ exports as a C compiler lays it out, and rad_yarn.h's C forms.
 *
 * A row is a struct's size (`field` empty) or one field's offset and size, in bytes. */
#ifndef ABI_LAYOUT_H
#define ABI_LAYOUT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AbiRow {
    const char* type;
    const char* field;
    size_t      offset;
    size_t      size;
} AbiRow;

extern const AbiRow abi_c_rows[];
extern const size_t abi_c_n_rows;

/* rad_yarn_frequency and rad_yarn_valid as C sees them, over a RadYarn of these four values. */
float abi_c_yarn_frequency(float factor, float low, float high, float magnitude, float inv, int i);
int   abi_c_yarn_valid(float factor, float low, float high, float magnitude);

#ifdef __cplusplus
}
#endif

#endif /* ABI_LAYOUT_H */
