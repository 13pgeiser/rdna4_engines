#include "hip/hip_runtime.h"
#include <cstdio>
#include <cstdlib>

/* ---- the context switch: callee-saved registers and the stack pointer (x86-64 SysV) */
extern "C" void emu_switch(void** save_sp, void* new_sp);
asm(R"(
.text
.globl emu_switch
.type emu_switch,@function
emu_switch:
    pushq %rbp
    pushq %rbx
    pushq %r12
    pushq %r13
    pushq %r14
    pushq %r15
    movq %rsp, (%rdi)
    movq %rsi, %rsp
    popq %r15
    popq %r14
    popq %r13
    popq %r12
    popq %rbx
    popq %rbp
    ret
)");

namespace {
constexpr size_t kStack = 512 * 1024;
struct Fiber {
    void* sp = nullptr;
    std::vector<char> stack;
    EmuIdx tid{};
    bool done = false;
    unsigned bar_gen = 0;      /* block barrier generation this fiber waits on */
    unsigned wbar_gen = 0;
    int waiting = 0;           /* 0 run, 1 block barrier, 2 wave barrier */
};
struct Sched {
    std::vector<Fiber> f;
    void* main_sp = nullptr;
    int cur = -1, nthreads = 0;
    EmuIdx bid{};
    void (*body)(void*) = nullptr;
    void* ctx = nullptr;
    /* block barrier */
    int arrive = 0; unsigned gen = 0;
    /* wave barriers and exchange slots, double-buffered by parity */
    int warrive[64] = {}; unsigned wgen[64] = {};
    int slot[64][2][32] = {};
    unsigned xparity[1024] = {};
} S;

void yield() { Fiber& me = S.f[S.cur]; emu_switch(&me.sp, S.main_sp); }

extern "C" void emu_trampoline() {
    S.body(S.ctx);
    S.f[S.cur].done = true;
    yield();
    std::abort();
}
/* the entry a fresh stack "returns" into, keeping the SysV alignment at call boundaries */
asm(R"(
.text
.globl emu_entry
.type emu_entry,@function
emu_entry:
    andq $-16, %rsp
    call emu_trampoline
    ud2
)");
extern "C" void emu_entry();
}  // namespace

EmuIdx& emu_thread_idx() { return S.f[S.cur].tid; }
EmuIdx& emu_block_idx() { return S.bid; }

void __syncthreads() {
    Fiber& me = S.f[S.cur];
    const unsigned g = S.gen;
    if (++S.arrive == S.nthreads) { S.arrive = 0; ++S.gen; return; }
    me.waiting = 1;
    while (S.gen == g) yield();
    me.waiting = 0;
}

static void wave_barrier(int w) {
    Fiber& me = S.f[S.cur];
    const unsigned g = S.wgen[w];
    if (++S.warrive[w] == 32) { S.warrive[w] = 0; ++S.wgen[w]; return; }
    me.waiting = 2;
    while (S.wgen[w] == g) yield();
    me.waiting = 0;
}

int emu_bpermute(int addr, int v) {
    const int t = (int)S.f[S.cur].tid.x, w = t >> 5, l = t & 31;
    const unsigned par = S.xparity[t]++ & 1u;
    S.slot[w][par][l] = v;
    wave_barrier(w);
    return S.slot[w][par][(addr >> 2) & 31];
}
int emu_readlane(int v, int lane) { return emu_bpermute(lane << 2, v); }
static short g_wa[64][2][32][8], g_wb[64][2][32][8];
emu_v8f emu_wmma_bf16(emu_v8s a, emu_v8s b, emu_v8f c) {
    const int t = (int)S.f[S.cur].tid.x, w = t >> 5, l = t & 31;
    const unsigned par = S.xparity[t]++ & 1u;
    for (int e = 0; e < 8; ++e) { g_wa[w][par][l][e] = a[e]; g_wb[w][par][l][e] = b[e]; }
    wave_barrier(w);
    auto bf = [](short v) { unsigned u = (unsigned)(unsigned short)v << 16; float f; std::memcpy(&f, &u, 4); return f; };
    const int N = l & 15, h = l >> 4;
    emu_v8f d;
    for (int e = 0; e < 8; ++e) {
        const int M = 8 * h + e;
        float acc = c[e];
        for (int k = 0; k < 16; ++k)
            acc += bf(g_wa[w][par][M + 16 * (k >> 3)][k & 7]) * bf(g_wb[w][par][N + 16 * (k >> 3)][k & 7]);
        d[e] = acc;
    }
    return d;
}
static _Float16 g_ha[64][2][32][8], g_hb[64][2][32][8];
emu_v8f emu_wmma_f16(emu_v8h a, emu_v8h b, emu_v8f c) {
    const int t = (int)S.f[S.cur].tid.x, w = t >> 5, l = t & 31;
    const unsigned par = S.xparity[t]++ & 1u;
    for (int e = 0; e < 8; ++e) { g_ha[w][par][l][e] = a[e]; g_hb[w][par][l][e] = b[e]; }
    wave_barrier(w);
    const int N = l & 15, h = l >> 4;
    emu_v8f d;
    for (int e = 0; e < 8; ++e) {
        const int M = 8 * h + e;
        float acc = c[e];
        for (int k = 0; k < 16; ++k)
            acc += (float)g_ha[w][par][M + 16 * (k >> 3)][k & 7] * (float)g_hb[w][par][N + 16 * (k >> 3)][k & 7];
        d[e] = acc;
    }
    return d;
}
unsigned emu_ballot(bool p) {
    unsigned m = 0;
    for (int l = 0; l < 32; ++l) m |= (unsigned)(emu_readlane(p ? 1 : 0, l) & 1) << l;
    return m;
}

int emu_grids = 0;
int emu_last_block = 0;
void emu_run_grid(dim3 grid, dim3 block, void (*body)(void*), void* ctx) {
    ++emu_grids;
    const int n = (int)(block.x * block.y * block.z);
    emu_last_block = n;
    if (block.y != 1 || block.z != 1 || n % 32 || n > 1024) { std::fprintf(stderr, "emu: block shape\n"); std::abort(); }
    S.body = body; S.ctx = ctx; S.nthreads = n;
    S.f.resize(n);
    for (auto& f : S.f) if (f.stack.size() != kStack) f.stack.assign(kStack, 0);
    for (unsigned bz = 0; bz < grid.z; ++bz)
    for (unsigned by = 0; by < grid.y; ++by)
    for (unsigned bx = 0; bx < grid.x; ++bx) {
        S.bid = EmuIdx{bx, by, bz};
        S.arrive = 0; S.gen = 0;
        for (int w = 0; w < 64; ++w) { S.warrive[w] = 0; S.wgen[w] = 0; }
        for (int t = 0; t < n; ++t) {
            Fiber& f = S.f[t];
            f.done = false; f.waiting = 0; f.tid = EmuIdx{(unsigned)t, 0, 0};
            S.xparity[t] = 0;
            /* a fresh stack: six callee-saved slots, then the entry as the return address */
            void** top = (void**)(f.stack.data() + kStack - 64);
            *--top = (void*)&emu_entry;
            for (int r = 0; r < 6; ++r) *--top = nullptr;
            f.sp = top;
        }
        int live = n;
        while (live > 0) {
            int progressed = 0;
            const int live_before = live;
            const unsigned g0 = S.gen; unsigned wg0[64]; for (int w = 0; w < 64; ++w) wg0[w] = S.wgen[w];
            int arr0 = S.arrive; int warr0[64]; for (int w = 0; w < 64; ++w) warr0[w] = S.warrive[w];
            live = 0;
            for (int t = 0; t < n; ++t) {
                if (S.f[t].done) continue;
                S.cur = t;
                emu_switch(&S.main_sp, S.f[t].sp);
                S.cur = -1;
                if (!S.f[t].done) ++live;
            }
            progressed = S.gen != g0 || S.arrive != arr0 || live != live_before;
            for (int w = 0; w < 64; ++w) progressed |= (S.wgen[w] != wg0[w]) || (S.warrive[w] != warr0[w]);
            if (live > 0 && !progressed) {
                int done = 0, wb = 0, bb = 0;
                for (auto& f : S.f) { done += f.done; wb += f.waiting == 2; bb += f.waiting == 1; }
                std::fprintf(stderr, "emu: DEADLOCK in block (%u,%u,%u): %d threads done, %d at a block "
                             "barrier, %d at a wave barrier -- a barrier not reached uniformly\n",
                             bx, by, bz, done, bb, wb);
                std::abort();
            }
        }
    }
}
