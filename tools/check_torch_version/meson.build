project('libtorch-version', 'cpp', default_options : ['default_library=static', 'cpp_std=c++14'])
torch_dep  = dependency('torch')
mnist_exe = executable('version', 'version.cpp', dependencies: torch_dep)


