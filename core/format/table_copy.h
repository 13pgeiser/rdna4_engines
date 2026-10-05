/* table_copy.h -- a container table copied into locked host memory (--ngram-placement ram).
 *
 * A COPY IN HUGE PAGES, NOT THE MAPPING LOCKED. Faulting a container's mapping in costs a fault every
 * 4 KiB page and leaves every row behind a 4 KiB translation, so a gather of scattered rows walks
 * the page tables on nearly every row; an anonymous MADV_HUGEPAGE region covers thousands of rows a
 * page. The region is locked as each page is first touched (MLOCK_ONFAULT), so nothing filled is
 * swapped out while the rest fills; it is read from the container by parallel positional reads, the
 * container's pages dropped behind them so the table is not held twice; and it is checked against
 * the mapping before anyone is handed it, because a wrong offset would serve every row from the
 * wrong place without an error. A host kernel reads an address no file backs as the memory it is
 * (libavx's table_file), so no plugin takes part. */
#pragma once
#include "radfile.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace rad {

struct TableCopy {
    void*  base  = nullptr;
    size_t bytes = 0;       /* the mapping's length: the table rounded up to 2 MiB */
};

/* Copies [src, src + n) of `file`'s mapping, which lies at file offset `foff`, into a new locked
 * region. RAD_OK with `out` filled, or a refusal with `why` saying what failed and nothing held. */
int  table_copy_make(const RadFile& file, const uint8_t* src, int64_t n, int64_t foff,
                     TableCopy* out, std::string* why);

/* Unlocks and unmaps it; a copy never made is a no-op. */
void table_copy_free(TableCopy* c);

}  /* namespace rad */
