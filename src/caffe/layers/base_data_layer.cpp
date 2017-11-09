#include <boost/thread.hpp>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

template <typename Dtype>
BaseDataLayer<Dtype>::BaseDataLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      transform_param_(param.transform_param()) {
}

template <typename Dtype>
void BaseDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (top.size() == 1) {
    output_labels_ = false;
  } else {
    output_labels_ = true;    // 只有一个标签的
  }
  data_transformer_.reset(
      new DataTransformer<Dtype>(transform_param_, this->phase_));    // 初始化数据变换器对象
  data_transformer_->InitRand();
  // The subclasses should setup the size of bottom and top
  DataLayerSetUp(bottom, top);
}

template <typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
      prefetch_(param.data_param().prefetch()),
      prefetch_free_(), prefetch_full_(), prefetch_current_() {
  for (int i = 0; i < prefetch_.size(); ++i) {
    prefetch_[i].reset(new Batch<Dtype>());   // 动态分配Batch缓存区,原来所指的对象会被销毁. 每次分配一个,这样速度会不会太慢
    prefetch_free_.push(prefetch_[i].get());    // prefetch_是一个共享指针类型,用get()获得传统 C 指针
  }   // BlockingQueue<Batch<Dtype>*> prefetch_free_;    // 空闲Batch队列
}   // 可以理解为数据库线程读取的数据首先放在prefetch_, 等读完prefetch_, 再取地址压入到网络读取数据的队列中. 这样就能达到快速且内存锁定数据填充, 防止读取冲突

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);

  // Before starting the prefetch thread, we make cpu_data and gpu_data
  // calls so that the prefetch thread does not accidentally make simultaneous
  // cudaMalloc calls when the main thread is running. In some GPUs this
  // seems to cause failures if we do not so.
  for (int i = 0; i < prefetch_.size(); ++i) {    //  在开区数据预取线程前,通过调用Blob相应函数进行cudaMalloc, 避免在多线程情况下同时进行导致CUDA API调用失败.
    prefetch_[i]->data_.mutable_cpu_data();   // 进行内存分配
    if (this->output_labels_) {
      prefetch_[i]->label_.mutable_cpu_data();
    }
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    for (int i = 0; i < prefetch_.size(); ++i) {
      prefetch_[i]->data_.mutable_gpu_data();
      if (this->output_labels_) {
        prefetch_[i]->label_.mutable_gpu_data();
      }
    }
  }
#endif
  DLOG(INFO) << "Initializing prefetch";
  this->data_transformer_->InitRand();
  StartInternalThread();    // 启动内部线程
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::InternalThreadEntry() {
#ifndef CPU_ONLY
  cudaStream_t stream;
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));    // 创建CUDA stream 非阻塞类型
  }
#endif

  try {
    while (!must_stop()) {    // 检查有没有线程中断请求
      Batch<Dtype>* batch = prefetch_free_.pop();
      load_batch(batch);    // batch是一个空的内存,通过调用load_batch以及数据库自带的游标,把batch填满
#ifndef CPU_ONLY
      if (Caffe::mode() == Caffe::GPU) {
        batch->data_.data().get()->async_gpu_push(stream);
        if (this->output_labels_) {
          batch->label_.data().get()->async_gpu_push(stream);   // 把Batch数据同步到GPU的Stream的消息中
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
#endif
      prefetch_full_.push(batch);   // 加入到带负载的Batch队列中
    }
  } catch (boost::thread_interrupted&) {    // 在哪里启动线程的? 捕获到异常,退出while循环
    // Interrupted exception is expected on shutdown
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamDestroy(stream));    // 销毁stream
  }
#endif
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_cpu(    // 把一个batch数据传入CPU内存中
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (prefetch_current_) {
    prefetch_free_.push(prefetch_current_);   // 把分配过的内存区拿去重新分配
  }
  prefetch_current_ = prefetch_full_.pop("Waiting for data");   // 从带负载的Batch队列中去除一个Batch对象
  // Reshape to loaded data.
  top[0]->ReshapeLike(prefetch_current_->data_);
  top[0]->set_cpu_data(prefetch_current_->data_.mutable_cpu_data());    // 把数据放到CPU可读的内存中
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(prefetch_current_->label_);
    top[1]->set_cpu_data(prefetch_current_->label_.mutable_cpu_data());
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BasePrefetchingDataLayer, Forward);
#endif

INSTANTIATE_CLASS(BaseDataLayer);
INSTANTIATE_CLASS(BasePrefetchingDataLayer);

}  // namespace caffe
