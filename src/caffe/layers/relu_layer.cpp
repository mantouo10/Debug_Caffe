#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"

namespace caffe {

template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();  // 每一层的特征节点
  Dtype* top_data = top[0]->mutable_cpu_data();     // 输出的特征节点,数据可变
  const int count = bottom[0]->count();     // 获取每层的节点数
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();      // 获取leaky Relu 的参数,默认0,即ReLu
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0))  // max 为模板形式, Leaky ReLu的公式
        + negative_slope * std::min(bottom_data[i], Dtype(0));
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)      // 链式法则,delta(y) = delta(kexi)*delta.
          + negative_slope * (bottom_data[i] <= 0));
    }
  }
}


#ifdef CPU_ONLY     //自己理解,也是一段重复的代码定义ReLULayer<Dtype>::Backward_gpu
STUB_GPU(ReLULayer);  //在device_alternate.hpp中，通过#ifdef CPU_ONLY定义了一些宏来取消GPU的调用：
#endif

INSTANTIATE_CLASS(ReLULayer); //使用抽象工厂的方式构建整个layer层，然后刚才的宏将layer注册成抽象的服务类，然后再使用的时候再向服务器提供者申请调用各个类，这样就可以把proto文件作为个各类的配置文件进行调用，这点和java的spring框架很像，使用的是SOA方式。
//         自己理解,其实就是一段程序替代,此程序为初始化层的类型
// #define INSTANTIATE_CLASS(classname) \
//    char gInstantiationGuard##classname; \
//    template class classname<float>; \
//    template class classname<double>
    //
}  // namespace caffe
