# SBF's Breakout code

In this directory:

-  Breakout/main.c is the self-contained host-only version (needs SDL)
-  SpiNN/bkout.c is SpiNNaker C code
-  SpiNN/bkout/main.c is the host visualiser (needs SDL)

_To run SpiNNaker version:_

ybug 192.168.240.253

boot scamp.boot spin3.conf

iptag 1 set . 17894

app_load  bkout.aplx @0,0 1 16

_Possibly useful:_

iobuf 1

app_stop 16
