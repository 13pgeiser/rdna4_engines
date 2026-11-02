/// \file test.cpp
/// \brief d3 NPU harness: System One requests through AutoDecisionModel, no HTTP.
///
///   test_d3_npu -m d3:lite -i requests.jsonl -o replies.jsonl [--reps N] [--warmup 1]
///
/// requests.jsonl: one object per line, {"id": ..., "body": "<raw JSON text>"} or {"id": ..., "request": {...}}.
/// replies.jsonl:  {"id", "status", "body", "ms"} per request, in input order. The same driver is the
///                 engine-level perf probe (latency per request without the HTTP stack).
#include <fstream>
#include <iostream>

#include "AutoDecisionModel/auto_decision_model.hpp"
#include "model_list.hpp"
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"

int main(int argc, char* argv[]) {
    arg_utils::po::options_description desc("Allowed options");
    arg_utils::po::variables_map vm;
    desc.add_options()("model,m", arg_utils::po::value<std::string>()->default_value("d3:lite"), "Decision model tag");
    desc.add_options()("input,i", arg_utils::po::value<std::string>()->required(), "requests.jsonl");
    desc.add_options()("output,o", arg_utils::po::value<std::string>()->required(), "replies.jsonl");
    desc.add_options()("reps", arg_utils::po::value<int>()->default_value(1), "run every request this many times (last reply kept)");
    desc.add_options()("warmup", arg_utils::po::value<bool>()->default_value(true), "warm up at load");
    arg_utils::po::store(arg_utils::po::parse_command_line(argc, argv, desc), vm);
    arg_utils::po::notify(vm);

    const std::string tag = vm["model"].as<std::string>();
    const int reps = vm["reps"].as<int>();
    std::string exe_dir = utils::get_executable_directory();
    std::string list_path = exe_dir + "/model_list.json";
    std::string models_dir = utils::get_models_directory();
    model_list models(list_path, models_dir);
    const std::string path = models.get_model_path(tag);
    nlohmann::json info = models.get_model_info(tag);

    auto device = flm_rt::device(0);
    auto t0 = time_utils::now();
    AutoDecisionModel model(&device, tag);
    model.load_model(path, info, false);
    auto t1 = time_utils::now();
    if (vm["warmup"].as<bool>()) model.warmup();
    auto t2 = time_utils::now();
    std::cerr << "load_ms," << time_utils::duration_ms(t0, t1).first << ",warmup_ms," << time_utils::duration_ms(t1, t2).first << "\n";

    std::ifstream in(vm["input"].as<std::string>());
    std::ofstream out(vm["output"].as<std::string>());
    std::string line;
    int n = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto rec = nlohmann::ordered_json::parse(line);
        const std::string body = rec.contains("body") ? rec["body"].get<std::string>() : rec["request"].dump();
        DecisionReply r;
        double ms = 0;
        for (int k = 0; k < reps; k++) {
            auto s = time_utils::now();
            r = model.system_one(body);
            ms = time_utils::duration_ms(s, time_utils::now()).first;
        }
        nlohmann::ordered_json o = {{"id", rec.value("id", std::to_string(n))}, {"status", r.status}, {"body", r.body}, {"ms", ms}};
        out << o.dump(-1, ' ', false, nlohmann::ordered_json::error_handler_t::replace) << "\n";
        n++;
    }
    std::cerr << "requests," << n << "\n";
    return 0;
}
