#include <vector>
#include <cfloat>
#include <iostream>

#include "caffe/layers/flex_conv_layer.hpp"
using namespace std;

namespace caffe {

template <typename Dtype>
__global__ void GpuGetMapping(const int nthreads, const Dtype* bottom_data,
  const int channels, const int height, const int width, const int pad_h,
  const int pad_w, const int sample_kernel_h, const int sample_kernel_w,
  Dtype* max_mapping, Dtype* min_mapping){
  CUDA_KERNEL_LOOP(index, nthreads){
    const int w = index % width;
    const int h = ( index / width ) % height;
    const int c = ( index / width / height ) % channels;
    const int n = ( index / width / height / channels );
    int hstart = h - pad_h;
    int wstart = w - pad_w;
    int hend = min(hstart + sample_kernel_h, height);
    int wend = min(wstart + sample_kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    Dtype maxval = -FLT_MAX;
    Dtype minval = FLT_MAX;
    const Dtype* bottom_slice =
      bottom_data + (n * channels + c ) * height * width;
    for (int ih = hstart; ih < hend; ++ih){
      for (int iw = wstart; iw < wend; ++iw){
        const int idx = ih * width + iw;
        // if (index == 0){
        //   printf("value = %f, index = %d\n",bottom_slice[idx], idx);
        // }

        if (bottom_slice[idx] > maxval){
          maxval = bottom_slice[idx];

        }
        if (bottom_slice[idx] < minval){
          minval = bottom_slice[idx];
        }
      }
    }
    max_mapping[index] = maxval;
    min_mapping[index] = minval;
    // if (index == 0){
    //   printf("maxvalue = %f\n", max_mapping[index]);
    //   printf("*******\n");
    // }
  }


}

template <typename Dtype>
__global__ void GpuComputeWeightMask(const int nthreads, const Dtype* weights,
    Dtype* max_bit_mask, Dtype* min_bit_mask){
      CUDA_KERNEL_LOOP(index, nthreads){
        if (weights[index] < 0.){
          min_bit_mask[index] = weights[index];
          max_bit_mask[index] = 0.;
        }else{
          min_bit_mask[index] = 0.;
          max_bit_mask[index] = weights[index];
        }
      }
}

template <typename Dtype>
__global__ void GpuComputeWeightDiff(const int nthreads, const Dtype* weight, Dtype* weight_diff,
    const Dtype* max_mask, const Dtype* min_mask){
      CUDA_KERNEL_LOOP(index, nthreads){
          if (weight[index] < 0.){
            weight_diff[index] = min_mask[index];
          }else{
            weight_diff[index] = max_mask[index];
          }
      }
}


template <typename Dtype>
void FlexConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const int wcount = this->blobs_[0]->count();



  Dtype *max_bit_mask, *min_bit_mask;
  CUDA_CHECK(cudaMalloc((void **) &max_bit_mask, wcount * sizeof(Dtype)));
  CUDA_CHECK(cudaMalloc((void **) &min_bit_mask, wcount * sizeof(Dtype)));


  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    const int bcount = bottom[i]->count();

    Dtype *max_mapping, *min_mapping;
    CUDA_CHECK(cudaMalloc((void **) &max_mapping, bcount * sizeof(Dtype)));
    CUDA_CHECK(cudaMalloc((void **) &min_mapping, bcount * sizeof(Dtype)));

    Dtype* top_data = top[i]->mutable_gpu_data();
    const int tcount = top[i]->count();
    const int images_count = top[i]->count(2);
    const int channels_count = top[i]->count(1);

    Dtype *max_top_data, *min_top_data;

    CUDA_CHECK(cudaMalloc((void **) &max_top_data, tcount * sizeof(Dtype)));
    CUDA_CHECK(cudaMalloc((void **) &min_top_data, tcount * sizeof(Dtype)));

    caffe_copy(tcount, top_data, max_top_data);
    caffe_copy(tcount, top_data, min_top_data);


    GpuGetMapping<Dtype><<<CAFFE_GET_BLOCKS(bcount), CAFFE_CUDA_NUM_THREADS>>>(
      bcount, bottom_data, this->channels_, height_, width_, pad_h_, pad_w_, sample_kernel_h_,
      sample_kernel_w_, max_mapping, min_mapping);


      // Dtype* maxcpu = (Dtype *)malloc(bcount * sizeof(Dtype));
      // Dtype* original = (Dtype *)malloc(bcount * sizeof(Dtype));
      // cudaMemcpy(maxcpu, max_mapping, bcount * sizeof(Dtype), cudaMemcpyDeviceToHost);
      // cudaMemcpy(original, bottom_data, bcount * sizeof(Dtype), cudaMemcpyDeviceToHost);
      // cout << "value: " << maxcpu[0] << endl;
      // for (int i = 0; i < 200; i++){
      //   if (maxcpu[0] == original[i]){
      //     cout << "index:" << i << endl;
      //   }
      // }
      // cout << maxcpu[0] << endl;
      // // cout << original[0] << " " << original[1] << " " << original[2] << " " <<
      // // original[32] << " " << original[34] << " " << original[64] << " " << original[65] << " " << original[66] << endl;
      // cout << original[0] << " " <<original[1] << " " << " " << original[2] << " " << original[3] << " " << endl;
      // cout << original[32] << " " <<original[33] << " " << " " << original[34] << " " << original[35] << " " << endl;
      // cout << "******" << endl;

    for (int n = 0; n < this->num_; ++n) {

      GpuComputeWeightMask<Dtype><<<CAFFE_GET_BLOCKS(wcount), CAFFE_CUDA_NUM_THREADS>>>(
        wcount, weight, max_bit_mask, min_bit_mask);

        // Dtype* maxcpu = (Dtype *)malloc(wcount * sizeof(Dtype));
        // Dtype* original = (Dtype *)malloc(wcount * sizeof(Dtype));
        // cudaMemcpy(maxcpu, max_bit_mask, wcount * sizeof(Dtype), cudaMemcpyDeviceToHost);
        // cudaMemcpy(original, weight, wcount * sizeof(Dtype), cudaMemcpyDeviceToHost);
        // cout << original[1] << endl;
        // cout << maxcpu[1] << endl;


      this->forward_gpu_gemm(max_mapping + n * this->bottom_dim_, max_bit_mask,
          max_top_data + n * this->top_dim_);
      this->forward_gpu_gemm(min_mapping + n * this->bottom_dim_, min_bit_mask,
          min_top_data + n * this->top_dim_);

      caffe_gpu_axpy<Dtype>(this->top_dim_, 1, max_top_data + n * this->top_dim_, min_top_data + n * this->top_dim_);
      int a = 0;
      // if (n == 0){
      //   Dtype* cpu = (Dtype *)malloc(tcount * sizeof(Dtype));
      //   cudaMemcpy(cpu, min_top_data, tcount * sizeof(Dtype), cudaMemcpyDeviceToHost);
      //   for (int i = 0; i < channels_count; i+=images_count){
      //     cout << "(" << cpu[i] << "," << cpu[i+1] << ")" << " ";
      //     a++;
      //   }
      //
      //   cout << "NUMS:" << a << endl;
      //
      // }

      caffe_copy(this->top_dim_, min_top_data + n * this->top_dim_, top_data + n * this->top_dim_);

      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
    cudaFree(max_mapping); cudaFree(min_mapping); cudaFree(max_top_data); cudaFree(min_top_data);
  }
  cudaFree(max_bit_mask); cudaFree(min_bit_mask);
}

template <typename Dtype>
void FlexConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const int wcount = reverse_dimensions() ? this->channels_ * this->blobs_[0]->count(1) :
                                            this->num_output_ * this->blobs_[0]->count(1);

  Dtype *max_mask, *min_mask;
  CUDA_CHECK(cudaMalloc((void **) &max_mask, wcount * sizeof(Dtype)));
  CUDA_CHECK(cudaMalloc((void **) &min_mask, wcount * sizeof(Dtype)));


  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();

  caffe_copy(wcount, weight_diff, max_mask);
  caffe_copy(wcount, weight_diff, min_mask);

  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();

    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();

      const int bcount = bottom[i]->count();
      Dtype *max_mapping, *min_mapping;
      CUDA_CHECK(cudaMalloc((void **) &max_mapping, bcount * sizeof(Dtype)));
      CUDA_CHECK(cudaMalloc((void **) &min_mapping, bcount * sizeof(Dtype)));

      GpuGetMapping<Dtype><<<CAFFE_GET_BLOCKS(bcount), CAFFE_CUDA_NUM_THREADS>>>(
        bcount, bottom_data, this->channels_, height_, width_, pad_h_, pad_w_, sample_kernel_h_,
        sample_kernel_w_, max_mapping, min_mapping);

      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {

          this->weight_gpu_gemm(max_mapping + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, max_mask);

          this->weight_gpu_gemm(min_mapping + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, min_mask);

          GpuComputeWeightDiff<Dtype><<<CAFFE_GET_BLOCKS(wcount), CAFFE_CUDA_NUM_THREADS>>>(
            wcount, weight, weight_diff, max_mask, min_mask);

          // this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
          //     top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
      cudaFree(max_mapping); cudaFree(min_mapping);
    }
  }
  cudaFree(max_mask); cudaFree(min_mask);
}

INSTANTIATE_LAYER_GPU_FUNCS(FlexConvolutionLayer);

}  // namespace caffe
