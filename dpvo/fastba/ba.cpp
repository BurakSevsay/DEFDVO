#include <torch/extension.h>
#include <vector>
#include <algorithm>


std::vector<torch::Tensor> cuda_ba(
    torch::Tensor poses,
    torch::Tensor patches,
    torch::Tensor intrinsics,
    torch::Tensor target,
    torch::Tensor weight,
    torch::Tensor lmbda,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor kk,
    int t0, int t1, int iterations);


std::vector<torch::Tensor> ba(
    torch::Tensor poses,
    torch::Tensor patches,
    torch::Tensor intrinsics,
    torch::Tensor target,
    torch::Tensor weight,
    torch::Tensor lmbda,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor kk,
    int t0, int t1, int iterations) {
  return cuda_ba(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations);
}


std::vector<torch::Tensor> neighbors(torch::Tensor ii, torch::Tensor jj)
{

  auto tup = torch::_unique(ii, true, true);
  torch::Tensor uniq = std::get<0>(tup).to(torch::kCPU);
  torch::Tensor perm = std::get<1>(tup).to(torch::kCPU);

  jj = jj.to(torch::kCPU);
  auto jj_accessor = jj.accessor<long,1>();

  auto perm_accessor = perm.accessor<long,1>();
  std::vector<std::vector<long>> index(uniq.size(0));
  for (int i=0; i < ii.size(0); i++) {
    index[perm_accessor[i]].push_back(i);
  }

  auto opts = torch::TensorOptions().dtype(torch::kInt64);
  torch::Tensor ix = torch::empty({ii.size(0)}, opts);
  torch::Tensor jx = torch::empty({ii.size(0)}, opts);

  auto ix_accessor = ix.accessor<long,1>();
  auto jx_accessor = jx.accessor<long,1>();

  for (int i=0; i<uniq.size(0); i++) {
    std::vector<long>& idx = index[i];
    std::stable_sort(idx.begin(), idx.end(),
       [&jj_accessor](size_t i, size_t j) {return jj_accessor[i] < jj_accessor[j];});

    for (int i=0; i < idx.size(); i++) {
      ix_accessor[idx[i]] = (i > 0) ? idx[i-1] : -1;
      jx_accessor[idx[i]] = (i < idx.size() - 1) ? idx[i+1] : -1;
    }
  }

  ix = ix.to(torch::kCUDA);
  jx = jx.to(torch::kCUDA);

  return {ix, jx};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &ba, "BA forward operator");
  m.def("neighbors", &neighbors, "temporal neighbor indices");
}
