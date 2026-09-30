import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

setup(
    name="xcalibur", version="0.1.0", packages=["xcalibur"],
    install_requires=["torch"], extras_require={"test": ["pytest"]},
    package_data={"xcalibur": ["*.cu", "supersonic/*.cu", "supersonic/*.inl"]},
    ext_modules=[CUDAExtension(
        "xcalibur._C", ["xcalibur/bindings.cu"],
        depends=["xcalibur/supersonic/" + f for f in ("topk.cu", "xR38F1.cu", "ptx.inl")],
        extra_compile_args={"cxx": ["-O3"], "nvcc": [
            "-O3", "-lineinfo", "-Xptxas=-v", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        ]},
    )],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
