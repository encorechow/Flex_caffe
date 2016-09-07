#include <vector>
#include <iostream>
#include "caffe/layers/flex_conv_layer.hpp"
using namespace std;

namespace caffe{

using std::min;
using std::max;


template <typename Dtype>
void FlexConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top){
    BaseConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
    FlexConvolutionParameter flexconv_param = this->layer_param_.flex_convolution_param();
    height_ = bottom[0]->shape(2);
    width_ = bottom[0]->shape(3);
    sample_kernel_h_ = flexconv_param.sample_kernel_h();
    sample_kernel_w_ = flexconv_param.sample_kernel_w();
    pad_h_ = sample_kernel_h_ / 2;
    pad_w_ = sample_kernel_w_ / 2;

}



template <typename Dtype>
void FlexConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void FlexConvolutionLayer<Dtype>::Get_Mapping(Blob<Dtype>* const& bottom, const Dtype* bottom_data,
  Dtype* max_mapping, Dtype* min_mapping){
  for (int c = 0; c < this->channels_; ++c){
    for (int h = 0; h < height_; ++h){
      for (int w = 0; w < width_; ++w){
          int hstart = h - this->pad_h_;
          int wstart = w - this->pad_w_;
          int hend = min(hstart + this->sample_kernel_h_, this->height_);
          int wend = min(wstart + this->sample_kernel_w_, this->width_);
          hstart = max(hstart, 0);
          wstart = max(wstart, 0);
          const int mapping_index = h * this->width_ + w;

          for (int ih = hstart; ih < hend; ++ih){
              for (int iw = wstart; iw < wend; ++iw){
                  const int index = ih * this->width_ + iw;

                  if (bottom_data[index] > max_mapping[mapping_index]){
                      max_mapping[mapping_index] = bottom_data[index];
                  }
                  if (bottom_data[index] < min_mapping[mapping_index]){
                      min_mapping[mapping_index] = bottom_data[index];
                  }
              }
          }
      }
    }
    bottom_data += bottom->offset(0, 1);
    max_mapping += bottom->offset(0, 1);
    min_mapping += bottom->offset(0, 1);

  }
}

template <typename Dtype>
void FlexConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data(); // blobs_[0] stands for weights.
  const int wcount = this->blobs_[0]->count();

  Dtype* max_bit_mask = new Dtype[wcount];
  Dtype* min_bit_mask = new Dtype[wcount];

  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    const int bcount = bottom[i]->count();

    Dtype* max_mapping = new Dtype[bcount];
    Dtype* min_mapping = new Dtype[bcount];

    Dtype* top_data = top[i]->mutable_cpu_data();
    const int tcount = top[i]->count();

    Dtype* max_top_data = new Dtype[tcount];
    Dtype* min_top_data = new Dtype[tcount];

    caffe_copy(tcount, top_data, max_top_data);
    caffe_copy(tcount, top_data, min_top_data);

    for (int n = 0; n < this->num_; ++n) {
      this->Get_Mapping(bottom[i], bottom_data + n * this->bottom_dim_ ,
        max_mapping + n * this->bottom_dim_, min_mapping + n * this->bottom_dim_);
        // cout << "Bottom: " << bottom_data[0] << " " << bottom_data[1] << " " << bottom_data[32] << endl;
        // cout << "max: " << max_mapping[0];


      // Compute weight bit mask
      for (int s = 0; s < wcount; ++s){
        if (weight[s] < 0.){
          min_bit_mask[s] = weight[s];
          max_bit_mask[s] = 0.;
        }else{
          min_bit_mask[s] = 0.;
          max_bit_mask[s] = weight[s];
        }
      }

      this->forward_cpu_gemm(max_mapping + n * this->bottom_dim_, max_bit_mask,
          max_top_data + n * this->top_dim_);
      this->forward_cpu_gemm(min_mapping + n * this->bottom_dim_, min_bit_mask,
          min_top_data + n * this->top_dim_);

      caffe_axpy<Dtype>(this->top_dim_, 1, max_top_data + n * this->top_dim_, min_top_data + n * this->top_dim_);

      caffe_copy(this->top_dim_, min_top_data + n * this->top_dim_, top_data + n * this->top_dim_);


      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data(); // blobs[1] stands for bias.
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void FlexConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();

  // Diff from convolution
  const int wcount = reverse_dimensions() ? this->channels_ * this->blobs_[0]->count(1) :
                                            this->num_output_ * this->blobs_[0]->count(1);

  Dtype* max_mask = new Dtype[wcount];
  Dtype* min_mask = new Dtype[wcount];

  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();

  caffe_copy(wcount, weight_diff, max_mask);
  caffe_copy(wcount, weight_diff, min_mask);
  /******************************/
  for (int i = 0; i < top.size(); ++i) {

    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();

    // Diff from convolution
    const int bcount = bottom[i]->count();
    Dtype* max_mapping = new Dtype[bcount];
    Dtype* min_mapping = new Dtype[bcount];
    /******************************/

    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {

        //Diff from convolution
        this->Get_Mapping(bottom[i], bottom_data, max_mapping, min_mapping);
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {

          //this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
          //    top_diff + n * this->top_dim_, weight_diff);
          this->weight_cpu_gemm(max_mapping + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, max_mask);

          this->weight_cpu_gemm(min_mapping + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, min_mask);

          for (int w = 0; w < wcount; w++){
            if (weight[w] < 0.){
              weight_diff[w] = min_mask[w];
            }else{
              weight_diff[w] = max_mask[w];
            }
          }
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(FlexConvolutionLayer);
#endif
INSTANTIATE_CLASS(FlexConvolutionLayer);

}
