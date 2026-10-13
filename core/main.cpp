/* main.cpp -- the entry point. Everything it does is in engine.h's comment; this file only turns
 * a command line into that sequence and a status into an exit code.
 */
#include "engine.h"
#include "hf_model.h"
#include "util/stallwatch.h"

int main(int argc, char** argv) {
    rad::Config cfg;
    int s = rad::config_parse(argc, argv, &cfg);
    if (s == 1) { rad::config_usage(argv[0]); return 0; }
    if (s < 0)  { rad::config_usage(argv[0]); return 2; }

    if (!rad::resolve_hf_model(cfg)) return 1;   /* a repo id: fetched, and --model points at it */

    rad::Engine engine(std::move(cfg));

    {
        /* A startup that stops printing says where it waits (util/stallwatch.h). Two minutes:
         * the slowest normal step, a container read from a slow disk, logs well inside that. */
        rad::StallWatch watch("startup", 120000);
        s = engine.start();
    }
    if (s < 0) {
        /* start() already said what and why; the report adds what it was running on and what it
         * had decided by then, which is what a log of a server that never came up is read for. */
        engine.print_startup_report(s);
        return 1;
    }

    if (engine.config().debug_graph) return engine.dump();
    s = engine.serve();
    if (s < 0) {
        /* A refusal between start() and listening (a chat template the defaults cannot render, a
         * thinking budget the model cannot hold) also ends a server that never came up. */
        if (!engine.startup_report_printed()) engine.print_startup_report(s);
        return 1;
    }
    return 0;
}
