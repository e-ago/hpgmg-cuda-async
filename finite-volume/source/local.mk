hpgmg-fv-y.c += $(call thisdir, \
	debug.c \
	timers.c \
	level.c \
	operators.fv4.c \
	mg.c \
	solvers.c \
	hpgmg-fv.c \
	)

hpgmg-fv-y.cc += $(call thisdir, \
	comm.cc \
	)

hpgmg-fv-y.cu += $(call thisdir, \
	cuda/operators.fv4.cu \
	)
