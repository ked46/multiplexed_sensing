'''Define functions to import and process the files listed above'''
import os
import pandas as pd
import numpy as np

# Some extraneous file names are included in this code for reference although they are not used in the analysis (for example the single_file is no longer used in analysis since the mixed culture data was sufficient for estimating parameters)

# Define the path to the directory containing the files
def define_metadata(community,plasmid,inhibitor):
    if community=='aTc_IPTG':
        path='Data files/'
        #list of file paths
        files=[path+'2023-03-23 mixed plate full timecourse.xlsx', path+'2023-03-24 mixed plate full timecourse.xlsx', path+'2023-03-23 crosstalk plus mixed plate full timecourse.xlsx', path+'2023-03-24 crosstalk plus mixed plate full timecourse.xlsx']#,
        #corresponding list of readers used (for conversion)
        readers=['J','loaner','loaner','J']
        #corresponding list of reporters used (for conversion)
        fluors=['GFP','mCh']
        fluor1=['GFP','GFP','GFP','GFP']
        fluor2=['mCh','mCh','mCh','mCh']
        #choose individual strain data excel file paths
        single_file='../2023-03-23 crosstalk plus mixed plate full timecourse.xlsx'
        #define sensor names listed in individual strain excel sheet
        sensor_names=['pLTetO-1','T5'] #in order of fp1 and fp2
        sensors=2
    elif community=='TTR_THS':
        path='Data files/'
        files=[path+'2023-12-04 mixed plate full timecourse_tecanJ_KD.xlsx',path+'2024-01-18 ttr ths mixed plate full timecourse_J_plate1.xlsx',path+'2024-01-18 ttr ths mixed plate full timecourse_loaner_plate2.xlsx'] #path+'2023-12-04 mixed plate full timecourse_loaner_AR.xlsx', (doesn't have 0,0 condition for subtraction
        #other files not used: '../../Data/Analyzed/2023-06-28 mixed plate full timecourse.xlsx','../../Data/Analyzed/2023-06-28 crosstalk plus mixed plate full timecourse.xlsx',
        readers=['J','J','loaner']#'loaner', #'loaner','nano'
        fluors=['YFP','CFP']
        fluor1=['YFP','YFP','YFP']#,'YFP']
        fluor2=['CFP','CFP','CFP']#,'CFP']
        single_file='../2023-06-28 crosstalk plus mixed plate full timecourse.xlsx'
        sensor_names=['ttr-yfp','ths-cfp'] 
        sensors=2
    elif community=='cuma_ohc_atc':
        path='Data files/'
        files= [path+'20240229 cuma ohc14 atc.xlsx',path+'20240224 cuma ohc14 atc.xlsx',path+'2023-07-14 corrected cuma atc ohc14 mixed plate full timecourse.xlsx'] 
        #other files not used: '../2023-07-14 corrected cuma atc ohc14 individual crosstalk plus mixed plate full timecourse.xlsx',
        #corresponding list of readers used (for conversion)
        readers=['nano','nano','loaner']#,'J'
        #corresponding list of reporters used (for conversion)
        fluors=['YFP','CFP','mCh']
        fluor1=['YFP','YFP','YFP']#,'YFP'
        fluor2=['CFP','CFP','CFP']#,'CFP'
        fluor3=['mCh','mCh','mCh']#,'mCh'
        #choose individual strain data excel file paths
        single_file='../2023-07-14 cuma ohc14 atc individual crosstalk plus mixed plate full timecourse.xlsx' 
        #define sensor names listed in individual strain excel sheet
        sensor_names=['cuma-yfp','ohc14-cfp','atc-mcherry'] #in order of fp1 and fp2
        sensors=3
    elif community=='van_dapg_nar':
        path='Data files/'
        files= [path+'20240531 nar van dapg.xlsx',path+'2025-02-21 nar van dapg mixed.xlsx']
        #corresponding list of readers used (for conversion)
        readers=['nano','nano']
        #corresponding list of reporters used (for conversion)
        fluors=['YFP','CFP','mCh']
        fluor1=['YFP','YFP']
        fluor2=['CFP','CFP']
        fluor3=['mCh','mCh']
        #choose individual strain data excel file paths
        single_file='../2024-10-16 van nari dapg individual crosstalk plate full timecourse.xlsx'
        #define sensor names listed in individual strain excel sheet
        sensor_names=['van-yfp','dapg-cfp','nar-mcherry'] #in order of fp1 and fp2
        sensors=3
    elif community=='antibiotic_data':#many of the variables are defined as None because the data is structured in a different way, so it skips some of the filtering and plate reader conversion steps which means it doesn't have certain variable metadata
        readers=None
        fluors=['GFP','BFP']
        fluor1=['GFP']
        fluor2=['BFP']
        single_file=None
        sensor_names=None
        sensors = 2
    elif community=='atc_van':
        path='sink water experiment analysis/'
        files=[path+'2024-12-08 atc-mch van-yfp sink ml experiment.xlsx',path+'2024-12-10 atc-mch van-yfp sink ml experiment.xlsx'] 
            #corresponding list of readers used (for conversion)
        readers=['J','J']#,'K'] #switch elmo for K when I add the actual full mixed experiment data
        #corresponding list of reporters used (for conversion)
        fluors=['mCh','YFP']
        fluor1=['mCh','mCh']
        fluor2=['YFP','YFP'] 
        #choose individual strain data excel file paths
        single_file=path+'2024-12-04 hospital sink week 12 atc van crosstalk.xlsx'
        #define sensor names listed in individual strain excel sheet
        sensor_names=['atc-mch','van-yfp'] #ensure they are in the same order as fp1 and fp2
        sensors=2
    elif community=='ttr_ths_ph':
        path='Data files/'
        files= [path+'20240531 ph ths ttr.xlsx']
        #corresponding list of readers used (for conversion)
        readers=['nano']
        #corresponding list of reporters used (for conversion)
        fluors=['YFP','CFP','mCh']
        fluor1=['YFP']
        fluor2=['CFP']
        fluor3=['mCh']
        #choose individual strain data excel file paths
        single_file=None
        #define sensor names listed in individual strain excel sheet
        sensor_names=['ttr-yfp','ths-cfp','ph-mcherry'] #in order of fp1 and fp2
        sensors=3
    else:
        print('community name chosen does not have any associated metadata available')
    if sensors<3:
        fluor3=None
    '''load appropriate files to create numpy arrays of all the raw readings'''
    if community=='antibiotic_data':
        path=f'Helenas data analysis/Data files/{plasmid}_{inhibitor}/'
        time_vector=np.load(f'{path}{plasmid}_{inhibitor}_time.npy')
        od_raws={}
        od_raws[0]=np.load(f'{path}{plasmid}_{inhibitor}_od.npy')
        fp1_raws={}
        fp1_raws[0]=np.load(f'{path}{plasmid}_{inhibitor}_gfp.npy')
        fp2_raws={}
        fp2_raws[0]=np.load(f'{path}{plasmid}_{inhibitor}_bfp.npy')
        fp3_raws=None
        #files for antibiotic data is different from regular data
        files=pd.read_csv(f'{path}{plasmid}_{inhibitor}_conditions.csv')
        #create a list of arrays to match the structure of the other community datasets, even though this set only has one array
        input_arrays={}
        input_arrays[0]=files[['[A]','[I]']].to_numpy(dtype=float)
        input_names=files.loc[0,['antibiotic_name','inhibitor_name']].to_numpy()
    else:
        #call function to import data
        od_dfs, fp1_dfs, fp2_dfs, fp3_dfs = import_exp_data(files, sensors)
        #filter dataframes to remove rows where 'Sensor' column is not equal to 'both' (blanks, individual strains, errors removed)
        filtered_od_dfs = filter_dataframes(od_dfs)
        filtered_fp1_dfs = filter_dataframes(fp1_dfs)
        filtered_fp2_dfs = filter_dataframes(fp2_dfs)
        if sensors==3:
            filtered_fp3_dfs = filter_dataframes(fp3_dfs)
        # Process the filtered DataFrames to create numpy arrays of raw data
        time_vector, od_raws, input_arrays, input_names = process_dataframes(community, filtered_od_dfs,sensors)
        _, fp1_raws, _, _ = process_dataframes(community, filtered_fp1_dfs,sensors)
        _, fp2_raws, _, _ = process_dataframes(community, filtered_fp2_dfs,sensors)
        if sensors==3:
            _, fp3_raws, _, _ = process_dataframes(community, filtered_fp3_dfs,sensors)
        else:
            fp3_raws=None
    return files, readers, fluors, fluor1, fluor2, fluor3, single_file, sensor_names, sensors,time_vector,od_raws,input_arrays,input_names,fp1_raws,fp2_raws,fp3_raws

#import each experimental dataset into a separate df
def import_exp_data(files,sensors):
    od_dfs = {}  # Dictionary to store od1df, od2df, and od3df
    fp1_dfs = {}  # Dictionary to store fp1df1, fp1df2, and fp1df3
    fp2_dfs = {}  # Dictionary to store fp2df1, fp2df2, and fp2df3
    fp3_dfs = {}  # Dictionary to store fp3df1, fp3df2, and fp3df3
    
    for ii,file_name in enumerate(files):
        file_path = f"{file_name}"
        file=pd.ExcelFile(file_path)
        od_dfs[ii] = pd.read_excel(file, 0)
        fp1_dfs[ii] = pd.read_excel(file,1)
        fp2_dfs[ii] = pd.read_excel(file,2)
        if sensors==3:
            fp3_dfs[ii] = pd.read_excel(file,3)
        else:
            fp3_dfs=None
    return od_dfs, fp1_dfs, fp2_dfs, fp3_dfs

#remove rows that are not all sensors co-cultured together (blank, individual strains, errors, etc)
def filter_dataframes(dfs):
    filtered_dfs = {}
    for key, df_dict in dfs.items():
        filtered_dfs[key] = df_dict[df_dict['Sensor'] == 'both']
    return filtered_dfs

# Common function to process DataFrames and create numpy arrays of raw data
def process_dataframes(community, filtered_dfs,sensors):
    #initialize arrays
    data_arrays = {}
    time_vector = None
    input_arrays={}
    input_cols=None
    max_valid_time=float('inf') #initialize the maximum timepoint measured

    #extract time vector from dataframes
    for key, df in filtered_dfs.items():
        #extract times listed in the columns starting with column labeled '0' until the last column
        df_time_columns=df.columns[np.where(df.columns==0)[0][0]:].to_numpy(dtype=float) 
        #update time_vector if this dataframe has less timepoints
        if time_vector is None or time_vector.shape[0] > df_time_columns.shape[0]:
            time_columns = df_time_columns
            time_vector = time_columns / 3600
        max_valid_time=min(max_valid_time, max(df_time_columns))
    #truncate time_columns and time_vector to the max valid timepoint
    valid_indices=time_columns<=max_valid_time
    time_columns=time_columns[valid_indices]
    time_vector=time_vector[valid_indices]
    #process each dataframe to align timepoints
    for key,df in filtered_dfs.items():
        #select columns corresponding with the time_vector (or those where the time lines up most closely for van/nar/dapg)
        data_columns = df.columns[np.where(df.columns==0)[0][0]:].to_numpy(dtype=float)
        closest_columns=[data_columns[np.argmin(np.abs(data_columns-t))] for t in time_columns]
        
        #select data from those columns to put into a numpy array
        data_arrays[key] = df[closest_columns].to_numpy(dtype=float)
        #for cuma_ohc_atc community, cut out data after t=20hr because there was a jump in plate reader readings
        if community=='cuma_ohc_atc':
            time_vector=time_vector[:72]
            data_arrays[key]=data_arrays[key][:,:72]
        #create input arrays
        if input_cols is None:
            input_cols=df.columns[df.columns.get_loc('Sensor')+1:df.columns.get_loc('Sensor')+sensors+1]
        input_arrays[key]=df[input_cols].to_numpy(dtype=float)

    return time_vector, data_arrays, input_arrays, input_cols


#convert data from different plate readers to match tecan J
def convert_reader_data(readers,fluors,raw_arrays,community):
    conv_arrays={}
    conv2J = pd.read_csv("Data files/reader conversion to J params.csv") 
    conv2J_new = pd.read_csv('Data files/2024 reader conversion to J params.csv')
    for i,j in enumerate(readers):
        #for older 2023 files they were done on the old calibration conversion
        if ((community=='aTc_IPTG')|((community=='cuma_ohc_atc')&(i>=2))): #add |((community=='TTR_THS')&(i<2)) in perentheses for ttr/ths community if using 6/28/23 data
            if j!='J':
                if fluors is None:
                    measurement='OD'
                else:
                    measurement=fluors[i]
                condition=conv2J[(conv2J['measurement'].str.contains(measurement)) & (conv2J['measurement'].str.contains(j))]
                conv_arrays[i]=raw_arrays[i]*(condition['b'].values[0])+(condition['c'].values[0])
            else:
                conv_arrays[i]=raw_arrays[i]
        #for the newer files they were done on the new calibration conversion
        else:
            if j!='J':
                if fluors is None:
                    measurement='OD'
                else:
                    measurement=fluors[i]
                condition=conv2J_new[(conv2J_new['measurement'].str.contains(measurement)) & (conv2J_new['measurement'].str.contains(j))]
                conv_arrays[i]=raw_arrays[i]*(condition['b'].values[0])+(condition['c'].values[0])
            else:
                conv_arrays[i]=raw_arrays[i]
    return conv_arrays

def subtract(community,time_vector,fp1_conv,fp2_conv,fp3_conv,sensors,input_arrays):
    basal_subtracted_fp1={}
    basal_subtracted_fp2={}
    basal_subtracted_fp3={}
        
    subtracted_fp1_conv_all=np.empty((0,len(time_vector)),dtype=float)
    subtracted_fp2_conv_all=np.empty((0,len(time_vector)),dtype=float)
    subtracted_fp3_conv_all=np.empty((0,len(time_vector)),dtype=float)
    
    for i, fluor in enumerate(fp1_conv):
        #subtract the basal expression condition from the data
        '''identify which row corresponds to zero inducers added'''
        basal_row_index=np.where(np.all(input_arrays[i]==0,axis=1))[0][0]
        basal_subtracted_fp1[i]=fp1_conv[i]-fp1_conv[i][basal_row_index]
        basal_subtracted_fp2[i]=fp2_conv[i]-fp2_conv[i][basal_row_index]
        if sensors>2:
            basal_subtracted_fp3[i]=fp3_conv[i]-fp3_conv[i][basal_row_index]
        subtracted_fp1_conv_all=np.vstack((subtracted_fp1_conv_all,basal_subtracted_fp1[i]))
        subtracted_fp2_conv_all=np.vstack((subtracted_fp2_conv_all,basal_subtracted_fp2[i]))
        if sensors>2:
            subtracted_fp3_conv_all=np.vstack((subtracted_fp3_conv_all,basal_subtracted_fp3[i]))
        else:
            subtracted_fp3_conv_all=None

    return subtracted_fp1_conv_all, subtracted_fp2_conv_all, subtracted_fp3_conv_all 

def normalize(subtracted_fp1_conv_all,subtracted_fp2_conv_all,subtracted_fp3_conv_all):
    # Find max and min fluor for each sensor fluor type
    max_value = np.max(subtracted_fp1_conv_all)
    min_value = np.min(subtracted_fp1_conv_all)
    range_value = max_value - min_value
    normalized_fp1_conv_all = (subtracted_fp1_conv_all - min_value) / range_value

    max_value = np.max(subtracted_fp2_conv_all)
    min_value = np.min(subtracted_fp2_conv_all)
    range_value = max_value - min_value
    normalized_fp2_conv_all = (subtracted_fp2_conv_all - min_value) / range_value
    
    if subtracted_fp3_conv_all is not None:
        max_value=np.max(subtracted_fp3_conv_all)
        min_value=np.min(subtracted_fp3_conv_all)
        range_value = max_value - min_value
        normalized_fp3_conv_all = (subtracted_fp3_conv_all - min_value) / range_value
    else: 
        normalized_fp3_conv_all=None
        
    return normalized_fp1_conv_all, normalized_fp2_conv_all, normalized_fp3_conv_all
        


