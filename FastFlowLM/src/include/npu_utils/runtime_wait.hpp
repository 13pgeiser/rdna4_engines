/// \file runtime_wait.hpp
/// \brief Backend-neutral validation for NPU command completion states.
#pragma once

#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace flm::npu {

// Stable ERT command-state values shared by the XRT and HRX adapters.
enum class ErtCmdState : int {
    New = 1,
    Queued = 2,
    Running = 3,
    Completed = 4,
    Error = 5,
    Abort = 6,
    Submitted = 7,
    Timeout = 8,
    NoResponse = 9,
    SkError = 10,
    SkCrashed = 11,
    Max = 12,
};

template <typename State>
constexpr ErtCmdState to_ert_cmd_state(State state) noexcept {
    return static_cast<ErtCmdState>(static_cast<int>(state));
}

constexpr std::string_view ert_cmd_state_name(ErtCmdState state) noexcept {
    switch (state) {
        case ErtCmdState::New: return "new";
        case ErtCmdState::Queued: return "queued";
        case ErtCmdState::Running: return "running";
        case ErtCmdState::Completed: return "completed";
        case ErtCmdState::Error: return "error";
        case ErtCmdState::Abort: return "abort";
        case ErtCmdState::Submitted: return "submitted";
        case ErtCmdState::Timeout: return "timeout";
        case ErtCmdState::NoResponse: return "noresponse";
        case ErtCmdState::SkError: return "skerror";
        case ErtCmdState::SkCrashed: return "skcrashed";
        case ErtCmdState::Max: return "max";
        default: return "unknown";
    }
}

inline void require_completed_ert_state(
    ErtCmdState state, std::string_view operation) {
    if (state == ErtCmdState::Completed) return;
    throw std::runtime_error(
        std::string(operation) + " failed: ERT command ended in state " +
        std::to_string(static_cast<int>(state)) + " (" +
        std::string(ert_cmd_state_name(state)) + ")");
}

template <typename Waitable>
auto wait_for_completion(Waitable& waitable, std::string_view operation) {
    if constexpr (std::is_void_v<decltype(waitable.wait())>) {
        // xrt::runlist::wait() returns void and is documented to throw when a
        // child fails. Some XRT versions return normally after the driver
        // publishes a terminal error state, so validate state() explicitly.
        waitable.wait();
        auto state = waitable.state();
        require_completed_ert_state(to_ert_cmd_state(state), operation);
        return state;
    } else {
        auto state = waitable.wait();
        require_completed_ert_state(to_ert_cmd_state(state), operation);
        return state;
    }
}

}  // namespace flm::npu
