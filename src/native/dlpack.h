#ifndef YOLOE_TENSORRT_DLPACK_H_
#define YOLOE_TENSORRT_DLPACK_H_

#include <cstdint>

typedef enum {
    kDLCPU = 1,
    kDLCUDA = 2,
} DLDeviceType;

typedef struct {
    DLDeviceType device_type;
    int device_id;
} DLDevice;

typedef struct {
    uint8_t code;
    uint8_t bits;
    uint16_t lanes;
} DLDataType;

typedef struct {
    void* data;
    DLDevice device;
    int ndim;
    DLDataType dtype;
    int64_t* shape;
    int64_t* strides;
    uint64_t byte_offset;
} DLTensor;

typedef struct DLManagedTensor {
    DLTensor dl_tensor;
    void* manager_ctx;
    void (*deleter)(struct DLManagedTensor* self);
} DLManagedTensor;

#endif
