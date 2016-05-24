External libraries dependencies are cfitsio and opencv:
Download the code from official webpage and build libraries in local machine.
Compress the outcome files and upload them to a ftp server. 
Take from there using wget and put in lib directory.

//to run the makefile, go to makefiles folder
make -f makefile

//to run in the console
g++ -o hello -L/mnt/project/lib -I/mnt/project/include HelloWorld.cpp -lopencv_core -Wl,-rpath=/mnt/project/lib


//for the makefile
gcc -o foo foo.c -L$(prefix)/lib -lfoo -Wl,-rpath=$(prefix)/lib


###Rules to build the entire project, first go to makefiles folder and then type:
make -f makefile all

###To clean the objects before rebuild the project:
make clean  




//To commit and add files to the repository do the following:
//#1 Clean binaries to make sure it only commit changes to modules
make clean
//#2 Add files commit to local
git commit -a -m "Commit message goes here"
//#3 Push your commit/messages to the host (github)
git push
