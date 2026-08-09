from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="aiop4090-fault-extension",
    ext_modules=[
        CUDAExtension(
            name="aiop4090_faults",
            sources=["bindings.cpp", "faults.cu"],
            extra_compile_args={
                "cxx": ["-O1"],
                "nvcc": ["-O1", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
