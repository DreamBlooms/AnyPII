// Physical-core introspection. See include/anypii/sys.hpp.
//
// std::thread::hardware_concurrency() reports logical CPUs, which on an SMT
// machine over-counts real cores by the thread-per-core factor and can leave
// llama.cpp oversubscribing shared execution resources. We count distinct
// (socket, core) pairs instead, then clamp to the logical CPUs this process may
// actually use (cgroups / affinity often restrict a container below the host
// topology).

#include "anypii/sys.hpp"

#include <algorithm>
#include <fstream>
#include <set>
#include <string>
#include <thread>

#if defined(__linux__)
    #include <sched.h>
#elif defined(_WIN32)
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#elif defined(__APPLE__)
    #include <sys/sysctl.h>
    #include <sys/types.h>
#endif

namespace anypii {
namespace {

int logical_cpu_count() {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        const int n = CPU_COUNT(&set);
        if (n > 0) return n;
    }
#endif
    const unsigned hc = std::thread::hardware_concurrency();
    return hc > 0 ? (int) hc : 1;
}

#if defined(__linux__)
int linux_physical_cores() {
    std::ifstream in("/proc/cpuinfo");
    if (!in) return 0;
    std::set<std::pair<std::string, std::string>> seen;
    std::string line, phys, core;
    bool have_phys = false;
    while (std::getline(in, line)) {
        if (line.rfind("physical id", 0) == 0) {
            const size_t colon = line.find(':');
            if (colon != std::string::npos) {
                phys = line.substr(colon + 1);
                have_phys = true;
            }
        } else if (line.rfind("core id", 0) == 0) {
            const size_t colon = line.find(':');
            if (colon != std::string::npos) {
                core = line.substr(colon + 1);
                // Some single-socket CPUs omit "physical id"; key on core alone then.
                seen.insert(have_phys
                                ? std::make_pair(phys, core)
                                : std::make_pair(std::string("0"), core));
                have_phys = false;
            }
        }
    }
    return (int) seen.size();
}
#endif

} // namespace

int physical_core_count() {
#if defined(__linux__)
    const int topo = linux_physical_cores();
    const int logical = logical_cpu_count();
    if (topo > 0) return std::max(1, std::min(topo, logical));
    return logical;
#elif defined(__APPLE__)
    int cores = 0;
    size_t len = sizeof(cores);
    if (sysctlbyname("hw.physicalcpu", &cores, &len, nullptr, 0) == 0 && cores > 0)
        return std::min(cores, logical_cpu_count());
    return logical_cpu_count();
#elif defined(_WIN32)
    // Count physical cores across all processor groups (excludes SMT siblings).
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len > 0) {
        std::string buf(len, '\0');
        auto * info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(&buf[0]);
        if (GetLogicalProcessorInformationEx(RelationProcessorCore, info, &len)) {
            int cores = 0;
            for (DWORD off = 0; off < len; off += info->Size) {
                info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(&buf[off]);
                if (info->Relationship == RelationProcessorCore) ++cores;
            }
            if (cores > 0) return std::min(cores, logical_cpu_count());
        }
    }
    return logical_cpu_count();
#else
    return logical_cpu_count();
#endif
}

} // namespace anypii
