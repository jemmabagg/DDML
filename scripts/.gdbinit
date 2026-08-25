add-auto-load-safe-path /home/baggjemm/DDML/scripts/.gdbinit

python
import sys
sys.path.insert(0, '/cvmfs/sw.hsf.org/contrib/x86_64-almalinux9-gcc11.4.1-opt/gcc/14.2.0-yuyjov/share/gcc-14.2.0/python')
from libstdcxx.v6.printers import register_libstdcxx_printers
register_libstdcxx_printers(None)
end
