#pragma once

#include "postprocess_common.h"
#include "tensor_views.h"
#include "visual_prompt_batch.h"

namespace yoloe_native {

void bind_native_main_runtime(py::module_& m);
void bind_native_visual_prompt_runtime(py::module_& m);
void bind_native_jetson_camera_source(py::module_& m);

} // namespace yoloe_native
