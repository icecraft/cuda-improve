
project('simple', 'cpp', 'cuda', version : '0.1.0')
incdir = include_directories('../Common')
cuda_dep = dependency('cuda', version : '>=10', modules : ['cublas'])
src = 'arraySum.cu'

executable('myexe', src, include_directories: incdir, dependencies: cuda_dep)


