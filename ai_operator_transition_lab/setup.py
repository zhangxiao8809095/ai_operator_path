from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="ai_operator_lab",
    ext_modules=[
        CUDAExtension(
            name="ai_operator_lab",
            sources=["src/bindings.cpp", "src/kernels.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
