saved cmd-lines:

    mm -C ~/Projects/barney/bin install && mm && gdb ./anariViewer -l barney


    mm && cp ./hayThereQT ./hayThereOffline ~/ && scp ./hayThereQT ./hayThereOffline wally: && /home/wald/opt/bin/mpirun -n 1 -host hasky /home/wald/hayThereQT /home/wald/barney/jets-2*.umesh : -n 1 -host wally /home/wald/hayThereOffline /home/wald/barney/jets-2*umesh 


lander, per-rank split:
    mm && BARNEY_METHOD=object-space ./hayThereQT ~/per-rank/lander-small-vort_mag-9000.1*umesh  ~/per-rank/lander-small-vort_mag-9000.2*umesh -xf hayThere.xf  ~/per-rank/lander-small-vort_mag-9000.3*umesh  --camera -41.3633 18.2164 22.2561 -41.3629 18.2159 22.2554 0 1 0 -fovy 60






dependencies on new machine (for building and/or mpirun)
	cmake-curses-gui freeglut3-dev libglfw3-dev libhwloc-dev libibverbs-dev net-tools openssh-server
