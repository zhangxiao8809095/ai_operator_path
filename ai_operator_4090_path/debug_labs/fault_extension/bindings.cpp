#include <torch/extension.h>

torch::Tensor fault_identity(torch::Tensor input);
torch::Tensor fault_out_of_bounds(torch::Tensor input);
torch::Tensor fault_shared_race(torch::Tensor input);
torch::Tensor fault_uninitialized_read(torch::Tensor input);
void fault_invalid_launch(torch::Tensor input);
void fault_illegal_address(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("identity", &fault_identity, "Safe identity baseline");
    module.def("out_of_bounds", &fault_out_of_bounds, "Intentional one-element OOB write");
    module.def("shared_race", &fault_shared_race, "Intentional shared-memory race");
    module.def("uninitialized_read", &fault_uninitialized_read,
               "Intentional read from uninitialized device allocation");
    module.def("invalid_launch", &fault_invalid_launch, "Intentional invalid block size");
    module.def("illegal_address", &fault_illegal_address,
               "Intentional asynchronous illegal-address execution fault");
}
