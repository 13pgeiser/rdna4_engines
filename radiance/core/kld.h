/* kld.h -- the KL divergence of this model against a recorded reference, at every scored position
 * of a fixed corpus.
 *
 * `--kld-record DIR` runs the corpus through the reference model (e.g. the bf16 checkpoint) once
 * and writes its full-vocabulary log-probabilities at every scored position. `--kld-ref DIR` runs a
 * candidate over the same tokens and compares its logits row by row, so each candidate costs one
 * prefill of the corpus.
 *
 * Reported: the distribution of per-position KL(P_ref || Q_candidate) (mean, median, 99th, 99.9th,
 * 99.99th percentile, max), top-1 agreement, and both models' perplexity on the corpus's next
 * tokens. The tail matters: a small mean KL can hide confidently wrong positions.
 *
 * The whole vocabulary, not a top-k: a truncated KL is a lower bound, loosest for flat
 * distributions, which are the high-KL tail.
 *
 * Reference files (DIR):
 *   kld.json     the manifest: the model, the corpus, every document's source, token count, first
 *                scored position and first row. Written last, so an interrupted run leaves a
 *                directory no --kld-ref reads.
 *   tokens.i32   every document's tokens, concatenated, so the candidate scores the reference's
 *                tokens rather than a re-tokenisation.
 *   logp.f16     [rows, n_vocab] log-probabilities, f16: finest near log p = 0, coarse only for
 *                tail tokens with little mass.
 *   top.bin      [rows, kKldTop] {int32 id, f32 log p}: the most likely tokens exactly, where a
 *                confident position's KL lives (f16 would cost ~1e-4 nats).
 *   rows.f32     [rows, 2] the reference's own next-token NLL and entropy.
 *
 * Document d scores positions [score_from, n - 1): the logits at p predict token p + 1, and the
 * last token predicts nothing in the corpus. Its row r is row0 + p - score_from.
 */
#pragma once
#include "rad_core.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rad {

class Vocab;

/* One scoring entry's rows of a step: request `id`'s prompt positions [pos0, pos0 + n) are rows
 * [row0, row0 + n) of the block the engine copied back. */
struct KldSpan {
    uint64_t id   = 0;
    int64_t  pos0 = 0;
    int64_t  n    = 0;
    int64_t  row0 = 0;
};

/* One rank's part of that block: its rows of `width` f32 logits, the vocabulary from `vocab0`. */
struct KldShard {
    const float* rows   = nullptr;
    int64_t      vocab0 = 0;
    int64_t      width  = 0;
};

/* The exact head kept beside every f16 row. */
static constexpr int kKldTop = 16;

class Kld {
public:
    Kld();
    ~Kld();

    /* Reads the corpus and creates DIR (record), or reads DIR's manifest and opens its rows (ref).
     * `model` names this run's model in the manifest and the report. */
    int open(const Config& cfg, std::shared_ptr<const Vocab> vocab, int64_t n_vocab,
             const std::string& model);

    /* open() split for already-tokenised documents (tests): add_doc() each, then record_into()
     * creates DIR; score_against() is the whole ref mode. */
    int add_doc(const std::string& source, const std::vector<int32_t>& ids, int64_t score_from);
    int record_into(const std::string& dir, int64_t n_vocab, const std::string& model);
    int score_against(const std::string& dir, const std::string& out, int64_t n_vocab,
                      const std::string& model);

    size_t n_docs() const { return docs_.size(); }
    /* Document d's tokens, and the request id the engine submits it under: d + 1. */
    std::vector<int32_t> tokens(size_t d) const;
    int64_t n_rows() const { return n_rows_; }
    int64_t rows_done() const { return done_n_.load(std::memory_order_relaxed); }

    /* A step's scored rows. Rows outside a document's scored range are skipped; a row seen twice
     * (re-prefill after preemption) is recomputed. Blocks until every row is written or scored,
     * since the engine reuses the block next step. */
    int score(const std::vector<KldSpan>& spans, const std::vector<KldShard>& shards);

    /* Record: checks every row was written and writes the manifest. Ref: the report, printed and,
     * with --kld-out, written. */
    int finish();

private:
    struct Doc {
        std::string source;
        int64_t     off = 0;          /* into tokens_ */
        int64_t     n = 0;
        int64_t     score_from = 0;
        int64_t     row0 = 0;
        int64_t     rows() const { return n - 1 - score_from; }
    };
    struct Job {
        int64_t row;                  /* the reference's row */
        int64_t at;                   /* the row in the copied block */
        int32_t next;                 /* the corpus's next token */
    };

    int  load_corpus(const Config& cfg, const std::shared_ptr<const Vocab>& vocab);
    int  load_ref(const std::string& dir);
    void size_rows();
    void record_row(const Job& j, const std::vector<KldShard>& sh, std::vector<uint16_t>& f16);
    void score_row(const Job& j, const std::vector<KldShard>& sh, std::vector<uint16_t>& f16);
    int  report();

    bool record_ = false;
    std::string dir_, out_, model_, ref_model_, corpus_;
    uint64_t corpus_hash_ = 0;
    int64_t n_vocab_ = 0;
    std::vector<int32_t> tokens_;
    std::vector<Doc> docs_;
    int64_t n_rows_ = 0;
    int fd_logp_ = -1, fd_top_ = -1, fd_rows_ = -1;

    /* Per reference row. `seen_` marks a row written or scored; the rest is the ref mode's. */
    std::vector<uint8_t> seen_;
    std::atomic<int64_t> done_n_{0};
    std::vector<float>   kl_, nll_q_, nll_p_;
    std::vector<uint8_t> top1_;
    std::atomic<int>     io_err_{0};
    float f16_lut_[65536];
};

}  /* namespace rad */
