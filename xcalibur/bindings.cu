#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <climits>
#include "supersonic/topk.cu"
#include "supersonic/xR38F1.cu"

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

torch::Tensor xR38F1(torch::Tensor W13, torch::Tensor X, torch::Tensor tKwi){
	check(W13, at::kInt, 3);
	check(X, at::kBFloat16, 2);
	check(tKwi, at::kInt, 2);
	TORCH_CHECK(W13.device() == X.device() && tKwi.device() == X.device(), "devices must match");
	int64_t E = W13.size(0), I = W13.size(1), H = W13.size(2), N = X.size(0), K = tKwi.size(1);
	TORCH_CHECK(N > 0 && N <= 65536 && E > 0 && E <= 65536, "1 <= N,E <= 65536 required");
	TORCH_CHECK(K > 0 && K <= 16 && K <= E, "1 <= K <= min(16,E) required");
	TORCH_CHECK(I > 0 && I <= INT_MAX - CTA && H > 0 && H <= INT_MAX / 4 && !(H & 7), "invalid I,H; H%8=0 required");
	TORCH_CHECK(X.size(1) == H && tKwi.size(0) == N, "X/routes dimensions must match W13");
	TORCH_CHECK(!(reinterpret_cast<uintptr_t>(W13.data_ptr()) & 15)
		&& !(reinterpret_cast<uintptr_t>(X.data_ptr()) & 15)
		&& !(reinterpret_cast<uintptr_t>(tKwi.data_ptr()) & 15), "16-byte alignment required");
	c10::cuda::CUDAGuard guard(X.device());
	auto Xs = torch::empty({E, 4 * H}, tKwi.options());
	auto Y = torch::empty({E, 8 + ((N + 7) / 8) * (8 + 32 * ((I + 7) / 8))}, tKwi.options());
	xR38F1_bf16<<<E, CTA, 0, at::cuda::getCurrentCUDAStream()>>>(
		reinterpret_cast<uint32_t*>(W13.data_ptr()), reinterpret_cast<uint32_t*>(X.data_ptr()),
		reinterpret_cast<uint32_t*>(Xs.data_ptr()), reinterpret_cast<uint32_t*>(Y.data_ptr()),
		reinterpret_cast<uint32_t*>(tKwi.data_ptr()), E, N, I, H, K
	);
	C10_CUDA_KERNEL_LAUNCH_CHECK();
	return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
	m.def("topk", &topk, pybind11::arg("logits"), pybind11::arg("K"), pybind11::arg("softmax") = true);
	m.def("xR38F1", &xR38F1, pybind11::arg("W13"), pybind11::arg("X"), pybind11::arg("routes"));
}
