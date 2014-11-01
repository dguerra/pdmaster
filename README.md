//to run the makefile, go to makefiles folder
make -f makefile

//to run in the console
g++ -o hello -L/mnt/project/lib -I/mnt/project/include HelloWorld.cpp -lopencv_core -Wl,-rpath=/mnt/project/lib


//for the makefile
gcc -o foo foo.c -L$(prefix)/lib -lfoo -Wl,-rpath=$(prefix)/lib
