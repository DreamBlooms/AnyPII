// System introspection shared by the runtime: chiefly the physical-core count
// used as the default CPU thread budget (SMT/hyperthreading siblings excluded).
#pragma once

namespace anypii {

// Number of physical CPU cores visible to this process (SMT siblings counted
// once), clamped to the logical CPUs the process may actually use. Falls back
// to std::thread::hardware_concurrency() when the topology cannot be read.
int physical_core_count();

} // namespace anypii
