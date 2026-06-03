#pragma once

#include <filesystem>

#include "model.hpp"

Model load_model(const std::filesystem::path& path);