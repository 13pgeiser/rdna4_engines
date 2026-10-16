/* issue_div_test.cpp -- the run phase's check of a DIV constraint on the ranged parameter.
 *
 * A band is resolved at its upper bound (core/plugin/bands.cpp), so a row with a DIV on the ranged
 * parameter wins a band whose top it divides and records the divisor; the issue path refuses a
 * value in the band it does not divide rather than launch the kernel outside its contract. The
 * Program is built by hand around one recording kernel, as runtime_test builds its own, and runs
 * on whichever device backend is linked.
 */
#include "rad_test.h"

#include "device/device.h"
#include "runtime/ctx.h"

#include <cstring>
#include <deque>
#include <functional>
#include <string>
#include <vector>

using rad::Ctx;
using rad::CtxDesc;
using rad::Program;

namespace {

std::vector<long long> g_launched;    /* the M each launch was issued for, read off the geometry */

int k_record(const RadArgs* a, RadStream) {
    long long m = -1;
    rad_args_geti(a, "M", &m);
    g_launched.push_back(m);
    return RAD_OK;
}

const RadParamSpec   g_params[] = { { "M", RAD_P_INT, RAD_REQUIRED, 0 } };
const RadOperandSpec g_opds[]   = { { "x", RAD_OPD_IN, 0 }, { "y", RAD_OPD_OUT, 0 } };
const RadOpSchema    g_schema   = { "t_div", g_params, 1, g_opds, 2, "a row with M % 16 == 0" };

std::function<void(Ctx&)> g_body;
void trampoline(RadCtx* c, const RadBatch*) { if (g_body) g_body(*static_cast<Ctx*>(c)); }

/* One op, one band up to 64, served by a row whose DIV on M is 16 (`div` 0: none). */
struct Fixture {
    Program       P;
    RadKernelInfo info{};
    rad::KernelRow row;
    Ctx           ctx;
    RadBatch      batch{};
    rad_buf       x = 0, y = 0;
    rad_op        op = 0;

    explicit Fixture(int64_t div) {
        P.meta.n_layers = 1;
        P.arena_bytes = 8192;
        P.scratch_bytes = 0;
        P.weights.resize(1);
        P.buffers.resize(1);
        P.ops.resize(1);
        for (const char* n : { "x", "y" }) {
            rad::BufferInfo b;
            b.name = n;
            b.decl.dtype = RAD_BF16;
            b.decl.rank = 2;
            b.decl.shape[0] = 64;
            b.decl.shape[1] = 4;
            b.decl.domain = RAD_DOMAIN_DEVICE;
            b.bytes = rad_dtype_bytes(RAD_BF16, 256);
            b.arena_offset = (int64_t)(P.buffers.size() - 1) * 1024;
            P.buffers.push_back(b);
        }
        x = 1;
        y = 2;
        info.name = "t_div_m16";
        info.op = "t_div";
        info.family = "test";
        info.domain = RAD_DOMAIN_DEVICE;
        info.launch = k_record;
        row.info = &info;
        row.plugin = "fake";
        rad::OpInfo o;
        o.op = "t_div";
        o.index = 1;
        o.ranged_key = "M";
        o.range_lo = 1;
        o.range_hi = 64;
        o.schema = &g_schema;
        rad::Band b;
        b.hi = 64;
        b.miss[RAD_DOMAIN_HOST] = "no host kernel";
        rad::Resolved& r = b.dom[RAD_DOMAIN_DEVICE];
        r.row = &row;
        r.geom.set_i("M", 64);
        r.ranged_div = div;
        o.bands.push_back(b);
        P.ops.push_back(o);
        op = 1;
    }
    int start() {
        CtxDesc d;
        d.program = &P;
        d.rank = 0;
        d.world_size = 1;
        d.arch_step = trampoline;
        return ctx.init(d);
    }
    int step(const std::function<void(Ctx&)>& body) {
        batch.phase = RAD_PHASE_DECODE;
        batch.n_tok = 1;
        batch.n_seq = 1;
        g_body = body;
        const int s = ctx.run_step(&batch);
        g_body = nullptr;
        return s;
    }
};

}  /* namespace */

/* A value the row's divisor divides launches it; one it does not is refused at issue, naming the
 * kernel and the divisor, and nothing launches. A row with no divisor serves every value of its
 * band. */
TEST(a_value_the_winning_rows_div_rejects_is_refused_at_issue) {
    {
        Fixture f(16);
        REQUIRE_EQ(f.start(), RAD_OK);
        g_launched.clear();
        CHECK_OK(f.step([&](Ctx& c) {
            for (int64_t n : { 16, 48, 64 }) {
                RadOperand opd[2] = { RAD_B(f.x), RAD_B(f.y) };
                CHECK_OK(c.issue(f.op, opd, 2, n));
            }
        }));
        CHECK_EQ(g_launched.size(), 3u);
        g_launched.clear();
        int st = RAD_OK;
        const std::string log = stderr_of([&] {
            st = f.step([&](Ctx& c) {
                RadOperand opd[2] = { RAD_B(f.x), RAD_B(f.y) };
                c.issue(f.op, opd, 2, 33);
            });
        });
        CHECK_EQ(st, RAD_E_SHAPE);
        CHECK(g_launched.empty());
        CHECK(has(f.ctx.step_error(), "t_div_m16"));
        CHECK(has(f.ctx.step_error(), "multiple of 16"));
        CHECK(has(log, "issued at 33"));
    }
    {
        Fixture f(0);
        REQUIRE_EQ(f.start(), RAD_OK);
        g_launched.clear();
        CHECK_OK(f.step([&](Ctx& c) {
            RadOperand opd[2] = { RAD_B(f.x), RAD_B(f.y) };
            CHECK_OK(c.issue(f.op, opd, 2, 33));
        }));
        CHECK_EQ(g_launched.size(), 1u);
    }
}

RAD_TEST_MAIN()
