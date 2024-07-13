COMMENT
Two state kinetic scheme synapse described by rise time tau1,
and decay time constant tau2. The normalized peak condunductance is 1.
Decay time MUST be greater than rise time.

The solution of A->G->bath with rate constants 1/tau1 and 1/tau2 is
 A = a*exp(-t/tau1) and
 G = a*tau2/(tau2-tau1)*(-exp(-t/tau1) + exp(-t/tau2))
	where tau1 < tau2

If tau2-tau1 -> 0 then we have a alphasynapse.
and if tau1 -> 0 then we have just single exponential decay.

The factor is evaluated in the
initial block such that an event of weight 1 generates a
peak conductance of 1.

Because the solution is a sum of exponentials, the
coupled equations can be solved as a pair of independent equations
by the more efficient cnexp method.

***
27 Apr 2016: move all ranges to one **NEURON bug** -> python can't read out RANGE variables that are defined in a second RANGE definition
L Goetz 
25 Aug 2016: add dt (ms) as PARAMETER
L Goetz & A Roth
12 Oct 2016: remove TABLE for mgblock
***

ENDCOMMENT

NEURON {
	POINT_PROCESS Exp2SynNMDA
	RANGE tau1, tau2, e, i, g, passive
	NONSPECIFIC_CURRENT i
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
}

PARAMETER {
	tau1 = 2 (ms) <1e-9,1e9>
	tau2 = 20 (ms) <1e-9,1e9>
	e=0	(mV)
	mg=1    (mM)		: external magnesium concentration

	dt (ms)
    	passive = 0 
}

ASSIGNED {
	v (mV)
	i (nA)
	g (uS)
	factor
}

STATE {
	A (uS)
	B (uS)
}

INITIAL {
	LOCAL tp
	if (tau1/tau2 > .9999) {
		tau1 = .9999*tau2
	}
	A = 0
	B = 0
	tp = (tau1*tau2)/(tau2 - tau1) * log(tau2/tau1)
	factor = -exp(-tp/tau1) + exp(-tp/tau2)
	factor = 1/factor
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	g = B - A
	i = g*mgblock(v)*(v - e)
}

DERIVATIVE state {
	A' = -A/tau1
	B' = -B/tau2
}

FUNCTION mgblock(v(mV)) {

	: from Jahr & Stevens
     if (passive == 0) {
	:printf("not frozen t=%g\n", t)
	mgblock = 1 / (1 + exp(0.062 (/mV) * -v) * (mg / 3.57 (mM)))
     } 
     if (passive == 1) {
	:printf("frozen t=%g\n", t)
        mgblock = 1 / (1 + exp(0.062 (/mV) * 80.0) * (mg / 3.57 (mM)))
     }
}

NET_RECEIVE(weight (uS)) {
	A = A + weight*factor
	B = B + weight*factor
}