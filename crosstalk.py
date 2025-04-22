import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def crosstalk(single_file,sensors,timepoint,sensor_names):
    #import excel files
    file=pd.ExcelFile(single_file)
    oddf=pd.read_excel(file,0)
    fp1df=pd.read_excel(file,1)
    fp2df=pd.read_excel(file,2)
    if sensors==3:
        fp3df=pd.read_excel(file,3)
    
    #convert time to hours and create time vector
    t_sec=oddf.columns[np.where(oddf.columns==0)[0][0]:].to_numpy(dtype=float)
    t=t_sec/3600
    timeloc=(np.abs(t-timepoint)).argmin() #finds location in time vector of that time
    
    #create the crosstalk dataframe of columns with sensor and input information
    crosstalk_df=fp1df[fp1df.Sensor!='both'].iloc[:,:sensors+1] 
    
    #define columns of dataframe that indicate the input columns (be careful later because, input_cols is something else)
    inputs_cols = crosstalk_df.columns[1:sensors+1] 
    
    #make input values numeric
    crosstalk_df[inputs_cols] = crosstalk_df[inputs_cols].apply(pd.to_numeric, errors='coerce')
    
    #select the fluorescence data starting at timeloc (with reference to the 'Time [s]' column)
    time_col_idx = np.where(oddf.columns == 'Time [s]')[0][0] + timeloc + 1
    crosstalk_df['fp1 raw'] = fp1df[fp1df.Sensor != 'both'].iloc[:, time_col_idx]
    crosstalk_df['fp2 raw'] = fp2df[fp2df.Sensor != 'both'].iloc[:, time_col_idx]
    if sensors == 3:
        crosstalk_df['fp3 raw'] = fp3df[fp3df.Sensor != 'both'].iloc[:, time_col_idx]
    
    # Subtract blank mean for each fluorescence protein (generalized for any number of sensors)
    for i in range(sensors):
        fp_col = f'fp{i+1}'
        raw_col = f'{fp_col} raw'
        crosstalk_df[fp_col] = crosstalk_df[raw_col] - crosstalk_df[crosstalk_df['Sensor'] == 'blank'][raw_col].mean()
    
    # Calculate fold change for each sensor over basal expression level (0 inducers added)    
    for i in range(sensors):
        fp_col = f'fp{i+1}'
        sensor_name = sensor_names[i]
        baseline = np.mean(crosstalk_df[fp_col][(crosstalk_df[inputs_cols] == 0).all(axis=1) & (crosstalk_df['Sensor'] == sensor_name)])
        crosstalk_df.loc[crosstalk_df['Sensor'] == sensor_name, 'fold'] = crosstalk_df[fp_col] / baseline
    
    # create mask to select rows that only have one inducer added
    mask = np.logical_or.reduce([
        (crosstalk_df[col] == crosstalk_df[col].max()) & (crosstalk_df[inputs_cols.difference([col])] == 0).all(axis=1)
        for col in inputs_cols
    ])
    #filter crosstalk dataframe according to mask    
    filterdf=crosstalk_df[mask]
    
    #compute mean fold values in a new array
    #initialize the empty array for crosstalk
    cross = np.zeros((len(inputs_cols), len(sensor_names)))
    
    for row_idx, excluded_col in enumerate(inputs_cols):
        for col_idx, sensor in enumerate(sensor_names):
            condition = (filterdf["Sensor"] == sensor) & (filterdf[inputs_cols.difference([excluded_col])] == 0).all(axis=1)
            cross[row_idx, col_idx] = np.mean(filterdf.loc[condition, 'fold'])
    
    # Create DataFrame for the heatmap
    crossdf = pd.DataFrame(cross, columns=sensor_names, index=inputs_cols)

    # Set Matplotlib to keep text as text in SVG files
    plt.rcParams['svg.fonttype'] = 'none'

    # plot fold change heatmap
    sns.set(font_scale=1.5)
    # Create the heatmap with vmin=min value of data and vmax set to the maximum value of the heatmap (can change the max to be the same as other sensors so they are on the same scale)
    vmax = crossdf.values.max()
    vmin=crossdf.values.min()
    fig,ax=plt.subplots()
    sns.heatmap(crossdf,cmap='Blues',vmin=vmin, vmax=vmax, annot=True, linewidths=0.5, linecolor='black', fmt='.2f', ax=ax)
    # Ensure all spines are visible
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    # Set spine color
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    plt.show()

    #calculate alpha parameter
    #numerators: transposed crosstalk array minus one 
    numerators = crossdf.T-1
    #denominators should be 1/(fold-1) only on the diagonal, only for intended inducer-sensor pairs
    denominators = np.eye(sensors)*np.where(numerators != 0, 1 / numerators, 0) #conditional statement is added to replace inf or nan with 0 in case that a value in numerators is 0 resulting in divide by zero
    # alpha should be sensors*inputs (whereas the heatmap is inputs*sensors)
    alphas=np.dot(denominators,numerators)

    #plot crosstalk matrix as actual alphas value rather than fold change value
    alphasdf=pd.DataFrame(alphas.T,columns=sensor_names,index=inputs_cols)
    fig,ax=plt.subplots()
    sns.heatmap(alphasdf,cmap='RdBu',vmin=-1, vmax=1,  annot=True, linewidths=0.5, linecolor='black', fmt='.2f', ax=ax)
    # Ensure all spines are visible
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    # Set spine color
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    plt.show()

    return alphas, fig


def mixed_crosstalk(file_name,sensors,timepoint,fluors):
    #import excel files
    file=pd.ExcelFile(file_name)
    oddf=pd.read_excel(file,0)
    fp1df=pd.read_excel(file,1)
    fp2df=pd.read_excel(file,2)
    if sensors==3:
        fp3df=pd.read_excel(file,3)
    
    #convert time to hours and create time vector
    t_sec=oddf.columns[np.where(oddf.columns==0)[0][0]:].to_numpy(dtype=float)
    t=t_sec/3600
    timeloc=(np.abs(t-timepoint)).argmin() #finds location in time vector of that time
    
    #create the crosstalk dataframe of columns with sensor and input information
    crosstalk_df=fp1df[(fp1df.Sensor=='both')|(fp1df.Sensor=='blank')].iloc[:,:sensors+1] 
    
    #define columns of dataframe that indicate the input columns (be careful because, input_cols is something else)
    inputs_cols = crosstalk_df.columns[1:sensors+1] 
    
    #make input values numeric
    crosstalk_df[inputs_cols] = crosstalk_df[inputs_cols].apply(pd.to_numeric, errors='coerce')
    
    #select the fluorescence data starting at timeloc (with reference to the 'Time [s]' column)
    time_col_idx = np.where(oddf.columns == 'Time [s]')[0][0] + timeloc + 1
    crosstalk_df['fp1 raw'] = fp1df[(fp1df.Sensor=='both')|(fp1df.Sensor=='blank')].iloc[:, time_col_idx]
    crosstalk_df['fp2 raw'] = fp2df[(fp1df.Sensor=='both')|(fp1df.Sensor=='blank')].iloc[:, time_col_idx]
    if sensors == 3:
        crosstalk_df['fp3 raw'] = fp3df[(fp1df.Sensor=='both')|(fp1df.Sensor=='blank')].iloc[:, time_col_idx]
    
    # Subtract blank mean for each fluorescence protein (generalized for any number of sensors)
    for i in range(sensors):
        fp_col = f'fp{i+1}'
        raw_col = f'{fp_col} raw'
        # #subtract mean blank value for each fluorescence type, but set the value to 0 if the subtraction results in a negative 
        # crosstalk_df[fp_col] = (crosstalk_df[raw_col] - crosstalk_df[crosstalk_df['Sensor'] == 'blank'][raw_col].mean()).clip(lower=0)
        crosstalk_df[fp_col] = crosstalk_df[raw_col] - crosstalk_df[crosstalk_df['Sensor'] == 'blank'][raw_col].mean()

    # Calculate fold change for each sensor over basal expression level (0 inducers added)    
    for i in range(sensors):
        fp_col = f'fp{i+1}'
        fold_col=f'fp{i+1} fold'
        # sensor_name = sensor_names[i]
        baseline = np.mean(crosstalk_df[fp_col][(crosstalk_df[inputs_cols] == 0).all(axis=1)]) #& (crosstalk_df['Sensor'] == sensor_name)])
        crosstalk_df[fold_col] = crosstalk_df[fp_col] / baseline
    
    # create mask to select rows that only have one inducer added
    mask = np.logical_or.reduce([
        (crosstalk_df[col] == crosstalk_df[col].max()) & (crosstalk_df[inputs_cols.difference([col])] == 0).all(axis=1)
        for col in inputs_cols
    ])
    #filter crosstalk dataframe according to mask    
    filterdf=crosstalk_df[mask]
    
    #initialize the empty array for crosstalk
    cross = np.zeros((len(inputs_cols), len(inputs_cols)))
    
    #compute mean fold values in a new array    
    for row_idx, excluded_col in enumerate(inputs_cols):
        for col_idx in range(sensors):
            fold_col=f'fp{col_idx+1} fold'
            condition = (filterdf[inputs_cols.difference([excluded_col])] == 0).all(axis=1) #& (filterdf["Sensor"] == sensor) 
            cross[row_idx, col_idx] = np.mean(filterdf.loc[condition, fold_col])
    
    # Create DataFrame for the heatmap
    crossdf = pd.DataFrame(cross, columns=fluors,index=inputs_cols)
    
    # Set Matplotlib to keep text as text in SVG files
    plt.rcParams['svg.fonttype'] = 'none'

    # plot fold change heatmap
    sns.set(font_scale=1.5)
    # Create the heatmap with vmin=min value of data and vmax set to the maximum value of the heatmap (can change the max to be the same as other sensors so they are on the same scale)
    vmax = crossdf.values.max()
    vmin=crossdf.values.min()
    fig,ax=plt.subplots()
    sns.heatmap(crossdf,cmap='Blues',vmin=vmin, vmax=vmax, annot=True, linewidths=0.5, linecolor='black', fmt='.2f', ax=ax)
    # Ensure all spines are visible
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    # Set spine color
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    plt.show()

    #calculate alpha parameter
    #numerators: transposed crosstalk array minus one 
    numerators = crossdf.T-1
    #denominators should be 1/(fold-1) only on the diagonal, only for intended inducer-sensor pairs
    denominators = np.eye(sensors)*np.where(numerators != 0, 1 / numerators, 0) #conditional statement is added to replace inf or nan with 0 in case that a value in numerators is 0 resulting in divide by zero
    # alpha should be sensors*inputs (whereas the heatmap is inputs*sensors)
    alphas=np.dot(denominators,numerators)

    #plot crosstalk matrix as actual alphas value rather than fold change value
    alphasdf=pd.DataFrame(alphas.T,columns=fluors,index=inputs_cols)
    fig,ax=plt.subplots()
    sns.heatmap(alphasdf,cmap='RdBu',vmin=-1, vmax=1,  annot=True, linewidths=0.5, linecolor='black', fmt='.2f', ax=ax)
    # Ensure all spines are visible
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    # Set spine color
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    plt.show()

    return alphas, fig

def mixed_crosstalk_simple(file_name,sensors,timepoint,fluors):
    #import excel files
    file=pd.ExcelFile(file_name)
    oddf=pd.read_excel(file,0)
    fp1df=pd.read_excel(file,1)
    fp2df=pd.read_excel(file,2)
    if sensors==3:
        fp3df=pd.read_excel(file,3)
    
    #convert time to hours and create time vector
    t_sec=oddf.columns[np.where(oddf.columns==0)[0][0]:].to_numpy(dtype=float)
    t=t_sec/3600
    timeloc=(np.abs(t-timepoint)).argmin() #finds location in time vector of that time
    
    #create the crosstalk dataframe of columns with sensor and input information
    crosstalk_df=fp1df[(fp1df.Sensor=='both')].iloc[:,:sensors+1] 
    
    #define columns of dataframe that indicate the input columns (be careful because, input_cols is something else)
    inputs_cols = crosstalk_df.columns[1:sensors+1] 
    
    #make input values numeric
    crosstalk_df[inputs_cols] = crosstalk_df[inputs_cols].apply(pd.to_numeric, errors='coerce')
    
    #select the fluorescence data starting at timeloc (with reference to the 'Time [s]' column)
    time_col_idx = np.where(oddf.columns == 'Time [s]')[0][0] + timeloc + 1
    crosstalk_df['fp1 raw'] = fp1df[(fp1df.Sensor=='both')|(fp1df.Sensor=='blank')].iloc[:, time_col_idx]
    crosstalk_df['fp2 raw'] = fp2df[(fp1df.Sensor=='both')|(fp1df.Sensor=='blank')].iloc[:, time_col_idx]
    if sensors == 3:
        crosstalk_df['fp3 raw'] = fp3df[(fp1df.Sensor=='both')|(fp1df.Sensor=='blank')].iloc[:, time_col_idx]
    
    # Subtract basal expression level (0 inducers added) mean for each fluorescence protein (generalized for any number of sensors)
    for i in range(sensors):
        fp_col = f'fp{i+1}'
        raw_col = f'{fp_col} raw'
        baseline = np.mean(crosstalk_df[raw_col][(crosstalk_df[inputs_cols] == 0).all(axis=1)]) #& (crosstalk_df['Sensor'] == sensor_name)])
        # #subtract mean blank value for each fluorescence type, but set the value to 0 if the subtraction results in a negative 
        # crosstalk_df[fp_col] = (crosstalk_df[raw_col] - crosstalk_df[crosstalk_df['Sensor'] == 'blank'][raw_col].mean()).clip(lower=0)
        crosstalk_df[fp_col] = crosstalk_df[raw_col] - baseline

    # # Calculate % activation for each sensor with respect to intended    
    # for i in range(sensors):
    #     fp_col = f'fp{i+1}'
    #     fold_col=f'fp{i+1} fold'
    #     # sensor_name = sensor_names[i]
    #     crosstalk_df[fold_col] = crosstalk_df[fp_col] / baseline
    
    # create mask to select rows that only have one inducer added
    mask = np.logical_or.reduce([
        (crosstalk_df[col] == crosstalk_df[col].max()) & (crosstalk_df[inputs_cols.difference([col])] == 0).all(axis=1)
        for col in inputs_cols
    ])
    #filter crosstalk dataframe according to mask    
    filterdf=crosstalk_df[mask]
    
    #initialize the empty array for crosstalk
    cross = np.zeros((len(inputs_cols), len(inputs_cols)))
    
    #compute mean fold values in a new array    
    for row_idx, excluded_col in enumerate(inputs_cols):
        for col_idx, fluor in enumerate(fluors):
            fold_col=f'fp{col_idx+1}'
            condition = (filterdf[inputs_cols.difference([excluded_col])] == 0).all(axis=1) #& (filterdf["Sensor"] == sensor) 
            cross[row_idx, col_idx] = np.mean(filterdf.loc[condition, fold_col])
    
    # Create DataFrame for the heatmap
    crossdf = pd.DataFrame(cross,columns=fluors, index=inputs_cols)
    
    # Set Matplotlib to keep text as text in SVG files
    plt.rcParams['svg.fonttype'] = 'none'

    # plot fold change heatmap
    sns.set(font_scale=1.5)
    # Create the heatmap with vmin=min value of data and vmax set to the maximum value of the heatmap (can change the max to be the same as other sensors so they are on the same scale)
    vmax = crossdf.values.max()
    vmin=crossdf.values.min()
    fig,ax=plt.subplots()
    sns.heatmap(crossdf,cmap='Blues',vmin=vmin, vmax=vmax, annot=True, linewidths=0.5, linecolor='black', fmt='.2f', ax=ax)
    # Ensure all spines are visible
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    # Set spine color
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    plt.show()

    #calculate alpha parameter
    #numerators: transposed crosstalk array so that fluor outputs are the rows and inputs are the columns
    numerators = crossdf.T
    #denominators should be 1/(fold-1) only on the diagonal, only for intended inducer-sensor pairs
    denominators = np.eye(sensors)*np.where(numerators != 0, 1 / numerators, 0) #conditional statement is added to replace inf or nan with 0 in case that a value in numerators is 0 resulting in divide by zero
    # alpha should be sensors*inputs (whereas the heatmap is inputs*sensors)
    alphas=np.dot(denominators,numerators)

    #plot crosstalk matrix as actual alphas value rather than fold change value
    alphasdf=pd.DataFrame(alphas.T,columns=fluors,index=inputs_cols)
    fig,ax=plt.subplots()
    sns.heatmap(alphasdf,cmap='RdBu',vmin=-1, vmax=1,  annot=True, linewidths=0.5, linecolor='black', fmt='.2f', ax=ax)
    # Ensure all spines are visible
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    # Set spine color
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    plt.show()

    return alphas, fig

def antibiotic_crosstalk(timepoint,time_vector,conditions,od_raws,fp1_raws,fp2_raws,fluors):
    timeloc=(np.abs(time_vector-timepoint)).argmin() #finds location in time vector of that time

    #create the crosstalk dataframe of columns with sensor and input information
    crosstalk_df=conditions[['[A]','[I]']].apply(pd.to_numeric,errors='coerce') 

    #define columns of dataframe that indicate the input columns (be careful because, input_cols is something else)
    inputs_cols=crosstalk_df.columns
    
    #select the fluorescence data starting at timeloc (with reference to the 'Time [s]' column)
    crosstalk_df['fp1 raw']=fp1_raws[0][:,timeloc]
    crosstalk_df['fp2 raw']=fp2_raws[0][:,timeloc]
    
    #this data doesn't have blanks so no blank subtraction is possible
    #crosstalk_df['fp1']=crosstalk_df['fp1 raw']-(crosstalk_df[crosstalk_df['Sensor']=='blank']['fp1 raw'].mean())
    #crosstalk_df['fp2']=crosstalk_df['fp2 raw']-(crosstalk_df[crosstalk_df['Sensor']=='blank']['fp2 raw'].mean())

    #calculate fold change for each fluorescent protein over 'basal expression level' (0 inducers added)
    crosstalk_df['fp1 fold']=crosstalk_df['fp1 raw']/np.mean(crosstalk_df['fp1 raw'][(crosstalk_df['[A]']==0)&(crosstalk_df['[I]']==0)])
    crosstalk_df['fp2 fold']=crosstalk_df['fp2 raw']/np.mean(crosstalk_df['fp2 raw'][(crosstalk_df['[A]']==0)&(crosstalk_df['[I]']==0)])

    # create mask to select rows that only have one inducer added
    mask= ((crosstalk_df['[A]']==crosstalk_df['[A]'].max())&(crosstalk_df['[I]']==0))|((crosstalk_df['[A]']==0)&(crosstalk_df['[I]']==crosstalk_df['[I]'].max()))
    #filter crosstalk dataframe according to mask 
    filterdf=crosstalk_df[mask]

    #initialize the empty array for crosstalk
    cross = np.zeros((len(inputs_cols), len(inputs_cols)))

    #compute mean fold values in a new array    
    for row_idx, excluded_col in enumerate(inputs_cols):
        for col_idx,fluor in enumerate(fluors):
            fold_col=f'fp{col_idx+1} fold'
            condition = (filterdf[inputs_cols.difference([excluded_col])] == 0).all(axis=1)
            cross[row_idx, col_idx] = np.mean(filterdf.loc[condition, fold_col])
    
    #extract specific input names
    input_names=conditions.loc[0,['antibiotic_name','inhibitor_name']].to_numpy()
    
    # Create DataFrame for the heatmap
    crossdf = pd.DataFrame(cross, columns=fluors, index=input_names)
    
    # Set Matplotlib to keep text as text in SVG files
    plt.rcParams['svg.fonttype'] = 'none'

    #plot fold change heatmap
    sns.set(font_scale=1.5)
    
    # Create the heatmap with vmin=min value of data and vmax set to the maximum value of the heatmap (can change the max to be the same as other sensors so they are on the same scale)    
    vmax = crossdf.values.max()
    vmin=crossdf.values.min()
    fig,ax=plt.subplots()
    sns.heatmap(crossdf,cmap='Blues',vmin=vmin, vmax=vmax, annot=True, linewidths=0.5, linecolor='black', fmt='.2f', ax=ax)
    # Ensure all spines are visible
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    # Set spine color
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    plt.show()
    
    #calculate alpha parameter
    #numerators: transposed crosstalk array minus one 
    numerators = crossdf-1
    #denominators should be 1/(fold-1) only on the diagonal, only for intended inducer-sensor pairs
    # denominators = np.eye(len(fluors))*np.where(numerators != 0, 1 / numerators, 0) #conditional statement is added to replace inf or nan with 0 in case that a value in numerators is 0 resulting in divide by zero
    denominators=np.max(np.abs(numerators),axis=0)
        
    # alpha should be sensors*inputs (whereas the heatmap is inputs*sensors)
    # alphas=np.dot(denominators,numerators)
    alphas=(numerators/denominators).T
    '''plot crosstalk matrix as actual alphas value rather than fold change value'''
    alphasdf=pd.DataFrame(100*alphas.T,columns=fluors,index=input_names)
    fig,ax=plt.subplots()
    sns.heatmap(alphasdf,cmap='RdBu',vmin=-100, vmax=100,  annot=True, linewidths=0.5, linecolor='black', fmt='.0f', ax=ax)
    # Ensure all spines are visible
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    # Set spine color
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    plt.show()

    return alphas, fig