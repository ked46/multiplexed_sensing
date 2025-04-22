# Calculate K and n from fitting single timepoint data with the hill function
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np


def dose_response_fitting(hill_timepoint, exp_inputs, normalized_fp1_conv_all, normalized_fp2_conv_all, normalized_fp3_conv_all, time_vector, input_names, sensors, fluors,community,plasmid,inhibitor):
    # Reset to Matplotlib defaults
    plt.style.use('default')  # Ensures Matplotlib is fully reset
    # Set Matplotlib to keep text as text in SVG files
    plt.rcParams['svg.fonttype'] = 'none'

    def hill_fun(x, r0, k, K, n):
        # return r0 + k*((x+1e-10)**n)/(((x+1e-10)**n)+(K**n))
        return r0 + k / (1 + (K / (x+1e-10))**n) #fgh:change x to (x+1e-10) if doing the fitting for calculating R2 (otherwise has a divide by 0) - n and K calculations will be slightly different

    #calculate the non zero min and max of the input values for setting the limits on the plots
    min_val=np.floor(np.log10(np.min(np.where(exp_inputs > 0, exp_inputs, np.inf))))
    max_val=np.round(np.log10(np.max(exp_inputs)))

    #create logspaced vector of input values for plotting hill function
    S=np.logspace(min_val-1,max_val+1,num=100)

    timeloc = (np.abs(time_vector - hill_timepoint)).argmin()  # defines location of that time

    K_calc=np.zeros(sensors*sensors)
    hill_calc=np.zeros(sensors*sensors)
    iteration=0

    fig, axes = plt.subplots(sensors, sensors, figsize=(sensors*3, sensors*2.5))
    axes=axes.flatten()

    for ii, array in enumerate([normalized_fp1_conv_all,normalized_fp2_conv_all,normalized_fp3_conv_all]):
        for jj, names in enumerate(input_names):
            column_mask = np.arange(exp_inputs.shape[1]) != jj
            mask = (exp_inputs[:,column_mask] == 0 ).all(axis=1) #&(exp_inputs[:,0]!=0) #mask for input2=0
            s1 = exp_inputs[:,jj][mask] #input 1 where input2=0
            if array is not None:
                fluor = array[mask, timeloc] #fluorescence values where input 2=0
                maxfluor = np.max(fluor)
                minfluor = np.min(fluor)
                rangefluor = maxfluor - minfluor
                med_input=np.median(s1)
                initial_guess = [minfluor, rangefluor, med_input, 1] #r0, k, K, n #original values were [minfluor,rangefluor,0.1,1]
                if ((plasmid=='Bla')&(inhibitor=='SUL')): #this is the only dataset which doesn't work with the upper bound for K being 1000, it is unable to find optimal parameters for amx-bfp likely because the data is flat
                    bounds=np.array([[minfluor, -1,0,0], #lower bounds
                                     [maxfluor,1,100,10]]) #upper bounds
                else:
                    bounds=np.array([[minfluor, -1,0,0], #lower bounds
                                     [maxfluor,1,1000,10]]) #upper bounds
                #old bounds options:
                # bounds=np.array([[0, -1,0,-10], #lower bounds
                #       [1,1,1000,10]]) #upper bounds
                # bounds=np.array([[minfluor, -rangefluor,0,-10], #lower bounds
                #   [maxfluor,rangefluor,1000,10]]) #upper bounds
                params, covariance = curve_fit(hill_fun, s1, fluor, p0=initial_guess,bounds=bounds)
                r0, k, K, n = params
                K_calc[iteration]=K
                hill_calc[iteration]=n
                fitted_curve = hill_fun(s1, r0, k, K, n)
                # r_squared = r2_score(fluor, fitted_curve)
        
                # Plot the data and the fitted curve in the respective subplot
                ax = axes[iteration]
                ax.plot(s1, fluor, 'o', color='blue', lw=2, label='Data')
                ax.plot(S, hill_fun(S, r0, k, K, n), 'r', label='Fitted Curve')
                ax.plot(K, hill_fun(K, r0, k, K, n), 'go', label='Km')
                ax.set_xscale('symlog', linthresh=10**min_val)
                #If you want all the y-axes to be on the same scale, otherwise comment out this line
                ax.set_ylim([-.05,1])
                # Only set x-axis label for the bottom row
                row_idx = iteration // sensors  # Determine row index
                if row_idx == sensors - 1:  # Last row
                    ax.set_xlabel(input_names[jj]) 
                else:
                    ax.set_xlabel('')
                ax.set_ylabel(fluors[ii] if jj==0 else '')  
                
                # Annotate r0, k, Km, and n values on the subplot
                if ii==jj:
                    x=0.03
                    y=0.9
                else:
                    x=0.03
                    y=0.9
                # else: 
                #     x=0.03
                #     y=0.13
                # ax.annotate(f'R2 = {r_squared:.3f}', xy=(0.05, 0.3), xycoords='axes fraction', fontsize=12, color='black')
                ax.annotate(f'Km = {K:.3f}', xy=(x, y), xycoords='axes fraction', fontsize=10, color='black')
                ax.annotate(f'n = {n:.2f}', xy=(x, y-0.1), xycoords='axes fraction', fontsize=10, color='black')
                #if you want the estimated r0 and k printed, uncomment these lines. These parameters are not used in subsequent analysis however.
                # ax.annotate(f'r0 = {r0:.2f}', xy=(x, y-0.2), xycoords='axes fraction', fontsize=10, color='black')
                # ax.annotate(f'k = {k:.2f}', xy=(x, y-0.3), xycoords='axes fraction', fontsize=10, color='black')
            iteration+=1
            
    # Adjust layout and show the subplots
    #plt.tight_layout()
    fig.suptitle(f'Timepoint {hill_timepoint} hours')
    plt.show()

    return K_calc, hill_calc, fig