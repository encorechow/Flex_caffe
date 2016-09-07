#ifndef CAFFE_FLEX_CONV_LAYER_HPP_
#define CAFFE_FLEX_CONV_LAYER_HPP_


#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_conv_layer.hpp"




namespace caffe{

template <typename Dtype>
class FlexConvolutionLayer : public BaseConvolutionLayer<Dtype> {
 public:
  explicit FlexConvolutionLayer(const LayerParameter& param)
      : BaseConvolutionLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "FlexConvolution"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Get_Mapping(Blob<Dtype>* const& bottom, const Dtype* bottom_data,
      Dtype* max_mapping, Dtype* min_mapping);


  virtual inline bool reverse_dimensions() { return false; }
  virtual void compute_output_shape();



private:

  int height_;
  int width_;
  //TO-DO: The parameters that are used to extract max-mapping and min-mapping
  int pad_h_; // padding height
  int pad_w_; // padding width
  int sample_kernel_h_; // extraction scale height
  int sample_kernel_w_; // extraction scale width


};
}

#endif
