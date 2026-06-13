from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="aiop4090",
    version="0.1.0",
    description="CUDA operator optimization lab for RTX 4090",
    package_dir={"": "src"},
    packages=find_packages("src"),
    ext_modules=[
        CUDAExtension(
            name="aiop4090._C",
            sources=[
                "src/aiop4090/csrc/bindings.cpp",
                "src/aiop4090/csrc/gemm.cu",
                "src/aiop4090/csrc/softmax.cu",
                "src/aiop4090/csrc/norm.cu",
                "src/aiop4090/csrc/attention.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
