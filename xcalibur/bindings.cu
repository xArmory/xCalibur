#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "supersonic/topk.cu"

static void check(const torch::Tensor& x, at::ScalarType dtype, int32_t dim){
	TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "expected contiguous CUDA tensor");
	TORCH_CHECK(x.scalar_type() == dtype && x.dim() == dim, "incorrect dtype or rank");
	TORCH_CHECK(!x.requires_grad(), "forward only; gradients are not implemented");
	auto p = at::cuda::getDeviceProperties(x.get_device());
	TORCH_CHECK(p->major == 8 && p->minor == 9, "SM89 (L4) required");
}

torch::Tensor topk(torch::Tensor logits, int64_t K, bool softmax){
	check(logits, at::kBFloat16, 2);
	int64_t N = logits.size(0), E = logits.size(1);
	TORCH_CHECK(N > 0 && N <= 65536 && E > 0 && E <= 65536, "1 <= N,E <= 65536 required");
	TORCH_CHECK(K > 0 && K <= 16 && K <= E, "1 <= K <= min(16,E) required");
	c10::cuda::CUDAGuard guard(logits.device());
	auto tKwi = torch::empty({N, K}, logits.options().dtype(at::kInt));
	auto stream = at::cuda::getCurrentCUDAStream();
	auto src = reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr());
	auto dst = reinterpret_cast<uint32_t*>(tKwi.data_ptr());
	if (softmax) topk_kernel<true><<<(N + 7) / 8, dim3(8, 4, 8), 0, stream>>>(src, dst, K, N, E);
	else topk_kernel<false><<<(N + 7) / 8, dim3(8, 4, 8), 0, stream>>>(src, dst, K, N, E);
	C10_CUDA_KERNEL_LAUNCH_CHECK();
	return tKwi;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
	m.def("topk", &topk, pybind11::arg("logits"), pybind11::arg("K"), pybind11::arg("softmax") = true);
}
