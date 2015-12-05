% This make.m is used under Windows

mex -glnx86 -v CFLAGS="\$CFLAGS --std=c99 -m32" -O -L/usr/lib32 -c svm.cpp
mex -cxx CFLAGS="\$CFLAGS --std=c99" -glnx86 -O -c svm_model_matlab.c
mex -glnx86 -cxx -v CFLAGS="\$CFLAGS --std=c99" -L/usr/lib32 -O svmtrain.c svm.o svm_model_matlab.o
mex CFLAGS='$CFLAGS --std=c99' -O svmpredict.c svm.o svm_model_matlab.o
mex CFLAGS='$CFLAGS --std=c99' -O libsvmread.c
mex CFLAGS='$CFLAGS --std=c99' -O libsvmwrite.c
