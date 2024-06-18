#Import general libraries (needed for functions)
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

# import the bayesian packages
import pymc3 as pm
import arviz as az

def get_GSP_counts(data, x_length, data_range):
#obtain the observed counts used in the bayesian model
#corrected for accomodation pooled data from 1Q, 2Q and 3Q interleave processes
    list_bitstring = ['0','00', '000', '100'] # all valid bistrings
    Y_list = []    
    for i_samples in data_range:
        row_list = []
        for c_index in range(x_length) :  
            total_counts = 0
            i_data = i_samples*x_length + c_index
            for key,val in data[i_data]['counts'].items():
                if  key in list_bitstring:
                    total_counts += val
            row_list.append(total_counts)
        Y_list.append(row_list)
    return np.array(Y_list)

# GSP plot

# prepare data for GSP plot
# get the calculated GSP values

def prepare_data_GSP_plot(my_model, my_trace, HDI = True):
    with my_model:
        hdi_prob = .94        
        theta_summary = az.summary(my_trace, round_to=12, hdi_prob = hdi_prob,
                                var_names = ["θ"], kind="stats")
    y1 = theta_summary.values[:,0]

    if HDI:
        # HDI values as bounds
        bounds_rmk = "(shown bounds are "+ str(int(100*hdi_prob)) + "% HDI)"
        y1_min = theta_summary.values[:,2]   
        y1_max = theta_summary.values[:,3]

    else:    
        # two SD bounds for plot
        bounds_rmk = "(shown bounds are ± two SD)"
        sy = theta_summary.values[:,1]
        y1_min = y1 - sy*2
        y1_max = y1 + sy*2
        
    return bounds_rmk, y1, y1_min, y1_max

def prepare_two_curves_GSP_plot(my_model, my_trace, X, Y,
                                HDI = True, hdi_prob  = .94):
    # prepare data for GSP plot
    # get the calculated GSP values + SD and HDI bounds
    # NB this retrievial is agnostic about 
    # the order of the reference and interleaved circuits
    
    c_len = int(X.shape[1]/2)
    
    with my_model:
        #  (hdi_prob=.94 is an usual default value for a credible interval)
        theta_summary = az.summary(my_trace, round_to=12, hdi_prob = hdi_prob,
                                var_names = ["θ"], kind="stats")

    theta = theta_summary.values.T

    t1_mask = np.tile(np.array(X[2],dtype=bool), theta.shape[0])
    theta1 = np.reshape(np.delete(theta,t1_mask),
             ((theta.shape[0],c_len) ))
    t2_mask = np.tile(np.array(X[1],dtype=bool), theta.shape[0])
    theta2 = np.reshape(np.delete(theta,t2_mask),
             ((theta.shape[0],c_len) ))

    y1 = theta1[0]
    y2 = theta2[0] 

    if HDI == True:
        # HDI values as bounds
        bounds_rmk = "(shown bounds are "+ str(int(100*hdi_prob)) + "% HDI)" 
        y1_min = theta1[2]
        y2_min = theta2[2]
        y1_max = theta1[3]
        y2_max = theta2[3]
    else:    
        # ± 2 SD bounds for plot
        bounds_rmk = "(shown bounds are ± two SD)"
        y1_min = y1 - 2*theta1[1]
        y1_max = y1 + 2*theta1[1]
        y2_min = y2 - 2*theta2[1]
        y2_max = y2 + 2*theta2[1]   

    # get the individual counts for the two curves standard and interleave
    Y1_mask = np.tile(np.array(X[2],dtype=bool), Y.shape[0])
    Y2_mask = np.tile(np.array(X[1],dtype=bool), Y.shape[0])
    Y1 = np.reshape(np.delete(Y,Y1_mask),((Y.shape[0],c_len)))
    Y2 = np.reshape(np.delete(Y,Y2_mask),((Y.shape[0],c_len)))
    
    return bounds_rmk, y1, y1_min, y1_max, y2, y2_min, y2_max, Y1, Y2     

def gsp_plot(scale, lengths, num_samples,shots, texto, title,
             y1, y1_min, y1_max, y2, y2_min, y2_max, Y1, Y2,
             first_curve = "Standard", second_curve = "Interleaved"):
            
    import matplotlib.pyplot as plt
    font = {'family' : 'DejaVu Sans',
            'weight' : 'normal',
            'size'   : 14}
    plt.rc('font', **font)
    fig, plt = plt.subplots(1, 1, figsize = [8,5])
    plt.set_yticks(np.arange(0.0,1.1,0.1))
    plt.set_ylim([0.95-scale, 1.01])
    plt.set_ylabel("P(0)", fontsize=16)
    plt.set_xlabel("Clifford Length", fontsize=16)

    plt.plot(lengths,y1,color="purple", marker="o", lw = 0.5)
    #plt.errorbar(lengths,y1,2*sy[0:m_len],
                 #color="purple", marker='o') #not used because not visible
    plt.fill_between(lengths, y1_min, y1_max,
                    alpha=.1, color = 'purple' ) 
    
    if second_curve != None:
        plt.plot(lengths,y2,color="cyan", marker='^', lw = 0.5)
        #plt.errorbar(lengths,y2,2*sy[m_len:2*m_len],
                     #color="cyan", marker='^') #not used because not visible
        plt.fill_between(lengths, y2_min, y2_max,
                        alpha=.1, color= 'cyan' )


    for i_sample in range(num_samples):
        plt.scatter(lengths, Y1[i_sample]/shots,
                   label = "data", marker="x",color="grey")
        if second_curve != None:
            plt.scatter(lengths, Y2[i_sample]/shots,
                   label = "data", marker="+",color="grey")

    legend_list = [first_curve]
    if second_curve != None :
        legend_list.append(second_curve)
    plt.legend(legend_list,
              loc = 'center right', fontsize=10)

    plt.text(0.25,0.95, texto, transform=plt.transAxes,
             verticalalignment='top',
             bbox=dict(facecolor='white'))
    plt.grid()
    plt.set_title(title,
                  fontsize=14)

# building the model
def create_model(T_test_values, X, Y, shots, scale,
                testval_s = 0.0001, upper_s = 0.001, lower_s = 0.00001, s_prior = "Uniform",
                alpha_Gamma = 1.5, beta_Gamma = 10000):
    my_model = pm.Model()
    with my_model:

        # Tying parameters
        BoundedUniform = pm.Bound(pm.Uniform,
                                   lower=np.fmax(T_test_values-0.1, np.full(T_test_values.shape,1.e-9)),
                                   upper=np.fmin(T_test_values+0.1, np.full(T_test_values.shape,1.-1e-9)))
        pi = BoundedUniform("Tying_Parameters",testval = T_test_values, shape = T_test_values.shape) 

        # sigma of Beta functions
        if s_prior == "Unif":
            sigma_t = pm.Uniform("σ_Beta",  testval = testval_s,
                                upper = upper_s, lower = lower_s, shape = Y.shape[1]) 
            
        elif s_prior == "Gamma":
            BoundedGamma = pm.Bound(pm.Gamma,
                                   lower= lower_s,
                                   upper= upper_s)
            sigma_t = BoundedGamma("σ_Beta", alpha = alpha_Gamma, beta = beta_Gamma,
                                    testval = testval_s, shape = Y.shape[1])           
        else: 
            raise Exception("Prior for sigma Beta only Uniform or Gamma at this time")
         
        if len(T_test_values) == 3: # standard RB
            # Tracing EPC
            EPC = pm.Deterministic('EPC', scale*(1-pi[1])) 
            # Tying function
            GSP = pi[0] * pi[1]**X + pi[2]
            
        elif len(T_test_values) == 4: # interleaved RB
            EPC = pm.Deterministic('EPC', scale*(1-pi[2])) 
            # Tying function
            GSP = pi[0] * ( X[1]*pi[1]**X[0] +\
                    X[2]*(pi[1]*pi[2])**X[0] ) +  pi[3]
        
        else: 
            raise Exception("Tying parameters must be 3 (standard) or 4 (interleaved)")
                            

        theta = pm.Beta('θ', mu=GSP, sigma = sigma_t,
                         shape = Y.shape[1])                                

        # Likelihood (sampling distribution) of observations    
        p = pm.Binomial("Counts", p = theta, observed = Y, n = shots)
    return my_model

# mean and sigma of EPC
def plot_epc(my_model, my_trace, epc_calib, epc_est_a,
             epc_est_a_err, epc_est_fm, epc_est_fm_err,
             epc_title):

    with my_model:
        az.plot_posterior(my_trace, var_names = ["EPC"],
                          round_to = 4, figsize = [10,6],
                          textsize = 12) 

    Bayes_legend =  "EPC  SMC: {0:1.3e} ± {1:1.3e}"\
                        .format(epc_est_a, epc_est_a_err)
    LSF_legend =    "EPC  LSF: {0:1.3e} ± {1:1.3e}".format(epc_est_fm, epc_est_fm_err)  
    Cal_legend =    "EPC Reference: {0:1.3e}".format(epc_calib)
    plt.axvline(x=epc_est_a,color='blue',ls="--")
    plt.axvline(x=epc_est_fm,color='orange',ls="-")    
    if epc_calib != np.nan:
        plt.axvline(x=epc_calib,color='r',ls="-", ymin = 0.01, ymax = 0.11, lw = 4)
    if epc_calib > 0.0:   
        plt.legend(("Posterior density", "$Highest\; density\; interval$ HDI",
                    Bayes_legend, LSF_legend,
                    Cal_legend), fontsize=12 )
    else: 
        plt.legend(("Posterior density", "$Highest\; density\; interval$ HDI",
                   Bayes_legend, LSF_legend), fontsize=12)
    plt.title(epc_title,fontsize=16); 
    
# perform reduced χ² calculation for Bayes hierarchical
def reduced_chisquare(ydata, sigma, my_trace): 
    mean_h = my_trace.posterior.mean(dim=['chain', 'draw'])
    theta_stacked = mean_h.θ.values
    r = ydata - theta_stacked
    chisq = np.sum((r / sigma) ** 2)
    NDF = len(ydata) - my_trace.posterior.dims['Tying_Parameters_dim_0']
    
    return chisq/NDF

def get_box_interleaved(my_summary, reduced_chisq):
    
    texto = "  alpha = {0:7.4f} ± {1:1.4e}"\
                .format(my_summary['mean']['Tying_Parameters[1]'],
                my_summary['sd']['Tying_Parameters[1]']) + "\n"                                          

    texto +=" alpha_c = {0:7.4f} ± {1:1.4e}"\
                .format(my_summary['mean']['Tying_Parameters[2]'],
                my_summary['sd']['Tying_Parameters[2]']) + "\n"                                             

    texto +="   EPC = {0:1.4e} ± {1:1.4e}"\
                .format(my_summary['mean']['EPC'],
                my_summary['sd']['EPC']) + "\n"

    texto +="             Fit χ² = {0:7.4f} "\
                .format(reduced_chisq)
    return texto
    