import numpy as np
import scipy as sp
from scipy import stats
exp = np.exp
sqrt = np.sqrt

#Lilliefors normality test, adpated from lillie.test from the R library nortest
#(Author(s): Juergen Gross, 
#webpage: http://rss.acs.unt.edu/Rdoc/library/nortest/html/lillie.test.html))
#input:
#   x : array
#output:
#   K: the test statistic
#	pvalue: the p-value of the test
def lilliefors(x): 
    x = np.array(x, dtype = np.double)
    x.sort()
    n = len(x)
    if (n < 5): 
    	exit()
    p = stats.norm.cdf((x - x.mean())/x.std(ddof= 1 ))
    Dplus = max(np.linspace(1, n, n)/n - p)
    Dminus = max(p - (np.linspace(1, n, n) - 1)/n)
    K = max(Dplus, Dminus)
    if n <= 100:
    	Kd = K
        nd = n
    else:
        Kd = K * ((n/100.0)**0.49)
        nd = 100
    pvalue = exp(-7.01256 * Kd**2 * (nd + 2.78019) + 2.99587 *
        Kd * sqrt(nd + 2.78019) - 0.122119 + 0.974598/sqrt(nd) +
        1.67997/nd)
    if pvalue > 0.1:
        KK = (sqrt(n) - 0.01 + 0.85/sqrt(n)) * K
        if KK <= 0.302:
        	pvalue = 1
        elif KK <= 0.5:
            pvalue = 2.76773 - 19.828315 * KK + 80.709644 * \
            	KK**2 - 138.55152 * KK**3 + 81.218052 * KK**4
        elif KK <= 0.9:
            pvalue = -4.901232 + 40.662806 * KK - 97.490286 * \
                KK**2 + 94.029866 * KK**3 - 32.355711 * KK**4
        elif KK <= 1.31:
            pvalue = 6.198765 - 19.558097 * KK + 23.186922 * \
                KK**2 - 12.234627 * KK**3 + 2.423045 * KK**4
        else:
        	pvalue = 0
    return K, pvalue
