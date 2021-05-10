project('libtorch-mnist', 'cpp', default_options : ['default_library=static', 'cpp_std=c++14'])
torch_dep  = dependency('torch')
mnist_exe = executable('mnist', 'mnist.cpp', dependencies: torch_dep)