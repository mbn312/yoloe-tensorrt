#include "native_common.h"

PYBIND11_MODULE(_native, m)
{
    m.doc() = "Native TensorRT runtime for yoloe_tensorrt";

    py::class_<yoloe_native::CudaTensorView>(m, "CudaTensorView")
        .def("__dlpack__", [](yoloe_native::CudaTensorView const& self, py::object stream) { return self.dlpack(std::move(stream)); }, py::arg("stream") = py::none())
        .def("__dlpack_device__", &yoloe_native::CudaTensorView::dlpack_device);

#ifdef YOLOE_TRT_ENABLE_JETSON_CAMERA
    py::class_<yoloe_native::CameraCudaTensorView>(m, "CameraCudaTensorView")
        .def("__dlpack__", [](yoloe_native::CameraCudaTensorView const& self, py::object stream) { return self.dlpack(std::move(stream)); }, py::arg("stream") = py::none())
        .def("__dlpack_device__", &yoloe_native::CameraCudaTensorView::dlpack_device);
#endif

    yoloe_native::bind_native_main_runtime(m);
    yoloe_native::bind_native_visual_prompt_runtime(m);
    yoloe_native::bind_native_jetson_camera_source(m);
}
