/* cvec_test.cpp -- the control vector's loader (core/cvec.h).
 *
 * The vector is the published package's format: a GGUF `controlvector` with one unit f32
 * `direction.<l>` per layer and three metadata keys. These cases write such a file, then check
 * both what the loader accepts and, just as important, what it refuses: a vector from another
 * architecture, one of the wrong width, and a layer range with a gap must stop startup rather than
 * steer a model along a direction that does not describe its residual stream.
 *
 * The compatibility question the port turns on is answered here rather than assumed: the file's
 * directions are 2560 wide and a Flash-Next container's stream is 2560, and the file states
 * `qwen4exp` as its architecture hint.
 */
#include "rad_test.h"

#include "cvec.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

/* A minimal GGUF writer, just enough for a control vector: the header, a string/int KV table, one
 * F32 one-dimensional tensor per direction, and the data blob at the file's alignment. */
struct GgufWriter {
    struct T { std::string name; std::vector<float> data; };
    std::vector<std::pair<std::string, std::string>> strs;
    std::vector<std::pair<std::string, int32_t>> ints;
    std::vector<T> tensors;

    static void u32(std::string& out, uint32_t v) {
        for (int i = 0; i < 4; ++i) out.push_back((char) ((v >> (8 * i)) & 0xff));
    }
    static void u64(std::string& out, uint64_t v) {
        for (int i = 0; i < 8; ++i) out.push_back((char) ((v >> (8 * i)) & 0xff));
    }
    static void str(std::string& out, const std::string& s) {
        u64(out, s.size());
        out += s;
    }

    void write(const std::string& path, uint64_t alignment = 32) const {
        std::string h;
        h += "GGUF";
        u32(h, 3);
        u64(h, tensors.size());
        u64(h, strs.size() + ints.size() + 1);   /* + general.alignment */
        for (const auto& [k, v] : strs) {
            str(h, k);
            u32(h, 8);   /* GGUF_STR */
            str(h, v);
        }
        for (const auto& [k, v] : ints) {
            str(h, k);
            u32(h, 5);   /* GGUF_I32 */
            u32(h, (uint32_t) v);
        }
        str(h, "general.alignment");
        u32(h, 4);   /* GGUF_U32 */
        u32(h, (uint32_t) alignment);

        uint64_t off = 0;
        std::string dir;
        for (const T& t : tensors) {
            str(dir, t.name);
            u32(dir, 1);                        /* n_dims */
            u64(dir, (uint64_t) t.data.size()); /* ne[0] */
            u32(dir, 0);                        /* ggml type 0 = F32 */
            const uint64_t padded = t.data.size() * 4;
            u64(dir, off);
            off += (padded + alignment - 1) / alignment * alignment;
        }
        h += dir;
        while (h.size() % alignment) h.push_back('\0');
        for (const T& t : tensors) {
            h.append((const char*) t.data.data(), t.data.size() * 4);
            while (h.size() % alignment) h.push_back('\0');
        }
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        f.write(h.data(), (std::streamsize) h.size());
    }
};

/* A vector with the published package's shape: `layers` directions of `n` values, each a unit
 * vector, plus the metadata its header carries. */
GgufWriter make_vector(int64_t n, int layers, const char* arch = "controlvector",
                       const char* hint = "qwen4exp") {
    GgufWriter w;
    w.strs.push_back({ "general.architecture", arch });
    w.strs.push_back({ "controlvector.model_hint", hint });
    w.ints.push_back({ "controlvector.layer_count", (int32_t) layers });
    for (int l = 1; l <= layers; ++l) {
        std::vector<float> d((size_t) n);
        for (int64_t i = 0; i < n; ++i) d[(size_t) i] = (float) std::sin(0.001 * (double) (l + 1) *
                                                                       (double) i + 0.1 * l);
        double ss = 0.0;
        for (float v : d) ss += (double) v * v;
        const float inv = (float) (1.0 / std::sqrt(ss));
        for (float& v : d) v *= inv;
        w.tensors.push_back({ "direction." + std::to_string(l), std::move(d) });
    }
    return w;
}

struct Loaded {
    rad::ControlVector cv;
    std::string err;
    int rc = 0;
};

/* A vector file, optionally with the package's JSON beside it (which is where the operation's
 * parameters live: there is no flag for them). */
Loaded load(const GgufWriter& w, int64_t n_embd, int64_t model_layers, const std::string& leaf,
            const std::string& json = std::string()) {
    const std::string path = radtest::tmp_path(leaf);
    w.write(path);
    if (!json.empty()) {
        const std::string side = path.substr(0, path.find_last_of('.')) + ".json";
        std::ofstream f(side, std::ios::binary | std::ios::trunc);
        f << json;
    }
    Loaded out;
    out.rc = rad::control_vector_load(path, "qwen4exp", n_embd, model_layers, &out.cv, &out.err);
    return out;
}

}  // namespace

TEST(the_published_shape_loads_and_is_normalised) {
    /* 47 directions of 2560 for a 48-layer model with the published package's JSON beside it:
     * its range, 4..44, steers. */
    const char* kPublished = R"({"mode":"project","scale":1.0,"layer_range":[4,44]})";
    const Loaded f = load(make_vector(2560, 47), 2560, 48, "flash-next.gguf", kPublished);
    CHECK_OK(f.rc);
    CHECK_EQ(f.cv.n_embd, (int64_t) 2560);
    CHECK_EQ(f.cv.first, 4);
    CHECK_EQ(f.cv.last, 44);
    /* Indexed by layer, so row 0 exists (the format has no direction for it) and 47 rows follow. */
    CHECK_EQ(f.cv.n_layers, (int64_t) 48);
    CHECK_EQ(f.cv.n_dirs, (int64_t) 47);

    for (int l = 1; l <= 47; ++l) {
        double ss = 0.0;
        for (int64_t i = 0; i < 2560; ++i)
            ss += (double) f.cv.dir[(size_t) (l * 2560 + i)] *
                  (double) f.cv.dir[(size_t) (l * 2560 + i)];
        CHECK_NEAR(std::sqrt(ss), 1.0, 1e-5);
        /* Unit rows, so the stored norm -- what llama.cpp keeps as `scales[il]` -- is 1 where the
         * range steers, and 0 outside it: the plugin declares an op exactly where it is nonzero. */
        CHECK_NEAR(f.cv.scales[(size_t) l], (l >= 4 && l <= 44) ? 1.0f : 0.0f, 1e-5);
    }
    /* Outside the range nothing is steered: no op, and the scale says so too. */
    CHECK_EQ(f.cv.covers(3), false);
    CHECK_EQ(f.cv.covers(4), true);
    CHECK_EQ(f.cv.covers(44), true);
    CHECK_EQ(f.cv.covers(45), false);

    /* The cache partition: nonzero with a vector loaded, and stable across loads of one file. */
    const Loaded again = load(make_vector(2560, 47), 2560, 48, "flash-next-2.gguf", kPublished);
    CHECK_OK(again.rc);
    CHECK(rad::control_vector_salt(f.cv) != 0);
    CHECK_EQ(rad::control_vector_salt(f.cv), rad::control_vector_salt(again.cv));
}

TEST(a_vector_for_another_architecture_is_refused) {
    /* A direction inferred from another model's residual basis is noise here. The hint is the one
     * place the file names its architecture, so it is checked when present. */
    const Loaded l = load(make_vector(2560, 47, "controlvector", "gemma4"), 2560, 48,
                          "wrong-arch.gguf");
    CHECK(l.rc < 0);
    CHECK(l.err.find("gemma4") != std::string::npos);
}

TEST(a_vector_of_the_wrong_width_is_refused) {
    /* 2048-wide directions in a 2560-wide model: the kernel would read past the row. */
    const Loaded l = load(make_vector(2048, 47), 2560, 48, "wrong-width.gguf");
    CHECK(l.rc < 0);
    CHECK(l.err.find("2560") != std::string::npos);
}

TEST(a_layer_range_with_a_gap_is_refused) {
    /* A vector with no direction for layer 30 while the range covers it would serve a model that
     * is silently unsteered on that layer -- a different model from the one the file describes. */
    GgufWriter w = make_vector(2560, 47);
    std::vector<GgufWriter::T> kept;
    for (auto& t : w.tensors)
        if (t.name != "direction.30") kept.push_back(t);
    w.tensors = std::move(kept);
    const Loaded l = load(w, 2560, 48, "gap.gguf");
    CHECK(l.rc < 0);
    CHECK(l.err.find("30") != std::string::npos);
}

TEST(not_a_control_vector_is_refused) {
    const Loaded l = load(make_vector(2560, 47, "qwen4exp", "qwen4exp"), 2560, 48,
                          "wrong-kind.gguf");
    CHECK(l.rc < 0);
    CHECK(l.err.find("controlvector") != std::string::npos);
}

/* The operation's parameters come from the package's own JSON, not a command line. A JSON that
 * states a range or scale is honoured; without one every direction the file carries steers at
 * scale 1; one that names another mode, or is not JSON, is refused by name. */
TEST(the_packages_json_sets_the_range_and_scale) {
    const Loaded a = load(make_vector(2560, 47), 2560, 48, "json-range.gguf",
                          R"({"mode":"project","scale":0.5,"layer_range":[8,20]})");
    CHECK_OK(a.rc);
    CHECK_EQ(a.cv.first, 8);
    CHECK_EQ(a.cv.last, 20);
    CHECK_NEAR(a.cv.scales[(size_t) 8], 0.5f, 1e-5);
    CHECK_EQ(a.cv.covers(7), false);
    CHECK_EQ(a.cv.covers(21), false);
    CHECK_EQ(a.cv.scales[(size_t) 7], 0.0f);
    CHECK_EQ(a.cv.scales[(size_t) 21], 0.0f);

    /* Another mode is not implemented. */
    const Loaded b = load(make_vector(2560, 47), 2560, 48, "json-add.gguf",
                          R"({"mode":"add","scale":0.1,"layer_range":[16,32]})");
    CHECK(b.rc < 0);
    CHECK(b.err.find("'add'") != std::string::npos);

    /* Not JSON. */
    const Loaded c = load(make_vector(2560, 47), 2560, 48, "json-broken.gguf", "{ not json");
    CHECK(c.rc < 0);
    CHECK(c.err.find("json-broken.json") != std::string::npos);

    /* No JSON: every direction the file carries, 1..47, at scale 1. No model's layer count picks
     * a range. */
    const Loaded d = load(make_vector(2560, 47), 2560, 48, "json-none.gguf");
    CHECK_OK(d.rc);
    CHECK_EQ(d.cv.first, 1);
    CHECK_EQ(d.cv.last, 47);
    CHECK_NEAR(d.cv.scales[(size_t) 1], 1.0f, 1e-5);
    CHECK_NEAR(d.cv.scales[(size_t) 47], 1.0f, 1e-5);
}

TEST(a_unit_vector_gives_scale_one_at_every_steered_layer) {
    const std::string path = radtest::tmp_path("scaled.gguf");
    make_vector(2560, 47).write(path);
    rad::ControlVector cv;
    std::string why;
    CHECK_OK(rad::control_vector_load(path, "qwen4exp", 2560, 48, &cv, &why));
    CHECK_NEAR(cv.scales[(size_t) 4], 1.0f, 1e-5);
    CHECK_NEAR(cv.scales[(size_t) 44], 1.0f, 1e-5);
}

/* Discovery is the branch's policy -- a file beside the model makes the capability available -- so
 * the order it searches in is part of the contract: the container's sidecar, then its directory,
 * then each `$RADIANCE_HOME/vectors`, with an empty file skipped rather than reported as found. */
TEST(discovery_finds_the_sidecar_then_the_directory_then_the_home) {
    const std::string dir   = radtest::tmpdir();
    const std::string model = dir + "/model.gguf";
    const std::string home  = dir + "/home";
    const std::string hvec  = home + "/vectors/qwen4exp-control-vector.gguf";
    std::string found, tried;

    /* Nothing anywhere: no answer, and the candidates are named for the log line. */
    CHECK_OK(rad::control_vector_discover(model, "qwen4exp", home, &found, &tried));
    CHECK(found.empty());
    CHECK(tried.find(dir + "/model.control-vector.gguf") != std::string::npos);

    /* A home's vectors/ entry is the last candidate, and an empty file there is not a vector (the
     * loader would refuse it), so discovery keeps looking rather than reporting it. */
    std::filesystem::create_directories(home + "/vectors");
    std::ofstream(hvec, std::ios::binary).close();
    CHECK_OK(rad::control_vector_discover(model, "qwen4exp", home, &found, &tried));
    CHECK(found.empty());
    make_vector(2560, 47).write(hvec);
    CHECK_OK(rad::control_vector_discover(model, "qwen4exp", home, &found, &tried));
    CHECK_EQ(found, hvec);

    /* The container's directory beats every home; the sidecar beats the directory. */
    make_vector(2560, 47).write(dir + "/control-vector.gguf");
    CHECK_OK(rad::control_vector_discover(model, "qwen4exp", home, &found, &tried));
    CHECK_EQ(found, dir + "/control-vector.gguf");
    make_vector(2560, 47).write(dir + "/model.control-vector.gguf");
    CHECK_OK(rad::control_vector_discover(model, "qwen4exp", home, &found, &tried));
    CHECK_EQ(found, dir + "/model.control-vector.gguf");

    /* Homes separated by `:` are searched in order: one without a vector defers to the next. */
    const std::string dir2 = dir + "/two";
    std::filesystem::create_directories(dir2 + "/vectors");
    make_vector(2560, 47).write(dir2 + "/vectors/qwen4exp-control-vector.gguf");
    CHECK_OK(rad::control_vector_discover(dir2 + "/m.gguf", "qwen4exp", dir + "/none:" + dir2,
                                          &found, &tried));
    CHECK_EQ(found, dir2 + "/vectors/qwen4exp-control-vector.gguf");
}

/* The range is the package's, and it must lie inside the container. Twenty directions fit a
 * 32-layer model; a JSON naming 4..40 does not. */
TEST(a_layer_range_past_the_model_is_refused) {
    const Loaded l = load(make_vector(2560, 20), 2560, 32, "range-past.gguf",
                          R"({"mode":"project","scale":1.0,"layer_range":[4,40]})");
    CHECK(l.rc < 0);
    CHECK(l.err.find("runs past") != std::string::npos);
}

/* A layer index the table cannot hold: a 48-layer model's table is rows 0..47, indexed by layer, so
 * direction.48 names one row past it -- as does a JSON range ending at 48. Both are refused rather
 * than writing or reading past the table. */
TEST(a_layer_at_the_models_own_count_is_refused) {
    const Loaded a = load(make_vector(2560, 48), 2560, 48, "layer-48.gguf");
    CHECK(a.rc < 0);
    CHECK(a.err.find("outside") != std::string::npos);

    const Loaded b = load(make_vector(2560, 47), 2560, 48, "range-to-48.gguf",
                          R"({"mode":"project","scale":1.0,"layer_range":[4,48]})");
    CHECK(b.rc < 0);
    CHECK(b.err.find("runs past") != std::string::npos);
}

RAD_TEST_MAIN()
