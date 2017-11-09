#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>

#ifdef USE_MKL
  #include "mkl.h"
#endif

#include "caffe/common.hpp"

namespace caffe {

// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
// The improvement in performance seems negligible in the single GPU case,
// but might be more significant for parallel training. Most importantly,
// it improved stability for large models on many GPUs.
inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda) {
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaMallocHost(ptr, size));    //如果是GPU模式,那么主机内存会以页锁定的方式锁定内存.这样分配好的内存不会被例如DMA这种内存存取机制动态占用，
    *use_cuda = true;
    return;
  }
#endif
#ifdef USE_MKL
  *ptr = mkl_malloc(size ? size:1, 64);  // MKL在资源利用和速度上比起Eigen还是有一定的优势的，矩阵越大越明显；但还是不如Matlab，可是Matlab用的也是MKL啊！！！
#else
  *ptr = malloc(size);
#endif
  *use_cuda = false;
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}

inline void CaffeFreeHost(void* ptr, bool use_cuda) {
#ifndef CPU_ONLY
  if (use_cuda) {
    CUDA_CHECK(cudaFreeHost(ptr));    // 释放内存
    return;
  }
#endif
#ifdef USE_MKL
  mkl_free(ptr);
#else
  free(ptr);
#endif
}


/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *
 * TODO(dox): more thorough description.
 */
class SyncedMemory {    // 负责存储分配以及主机和设备间的同步
 public:
  SyncedMemory();
  explicit SyncedMemory(size_t size);
  ~SyncedMemory();
  const void* cpu_data();
  void set_cpu_data(void* data);    // cpu_ptr_所指向的内存释放，并且cpu_ptr_指向入参data所指向内存
  const void* gpu_data();
  void set_gpu_data(void* data);
  void* mutable_cpu_data();   // 返回分配的cpu的内存地址:cpu_ptr_, 置状态为head_ = HEAD_AT_CPU
  void* mutable_gpu_data();
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };    // 状态机变量:未初始化/CPU数据有效/GPU数据有效/已同步
  SyncedHead head() { return head_; }   // 获得当前的状态
  size_t size() { return size_; }   // 获得当前的存储空间

#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);    //异步传输数据，将数据从cpu拷贝到gpu
#endif

 private:
  void check_device();

  void to_cpu();    // 数据同步到CPU
  void to_gpu();
  void* cpu_ptr_;
  void* gpu_ptr_;
  size_t size_;
  SyncedHead head_;
  bool own_cpu_data_;
  bool cpu_malloc_use_cuda_;
  bool own_gpu_data_;   // 是否有gpu内存
  int device_;    // GPU的设备ID号

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
