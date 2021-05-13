import scikit_posthocs as sp
import scipy.stats as st
import numpy as np
import pandas as pd

# runs kruskall-wallis test and then conover post hoc test. Good for non normal
# distributions
def run_kruskall_posthoc(x,y,df):
    # select only parts of df that are needed for analysis and remove outliers 
    df=remove_outliers(x,y,df)
    # Post warning if df is less than 5 samples
    if len(df)<5:
        print('Warning: Sample Size Smaller Than 5. Kruskall Wallace Test Value Suspect')
    # Reorded dataframes and run sruskall wallace test
    groups=[]
    group_names=df[x].drop_duplicates()
    group_names=np.array(group_names.values)
    for i in range(len(group_names)):
        indv_group=df[df[x]==group_names[i]][y].values
        indv_group=indv_group.tolist()
        groups.append(indv_group)
    # run kruskall-wallis test 
    k=st.mstats.kruskalwallis(*groups)
    # if kruskall-wallace test is satisfied then run post hoc tests, and return results as a dataframe
    # run post hoc dunn following kruskall wallace test 
    # this returns p values of each group with the diagonals being -1 beucase 
    # you're comparing the same group
    x=sp.posthoc_conover(df,val_col=y,group_col=x,p_adjust='holm')
    # Store the result of comparisons of different groups in dataframe
    x['Measurement']=[y]*len(x)
    x['Comparison Group']=x.index.values
    x=x.reset_index()
    x['Krusal-Wallis p value']=[k[1]]*len(x)
    return x



# method for removing outliers using interquartile range
def remove_outliers(x,y,df):
    group_names=df[x].drop_duplicates()
    group_names=np.array(group_names.values)
    # dataframe that we'll store the filtered results in
    result_df=pd.DataFrame()
    # removing outliers for each group
    for i in range(len(group_names)):
        loop_df=pd.DataFrame()
        sub_df=df[df[x]==group_names[i]][y]
        sub_df=sub_df.dropna()
        # calculate the 25% and 75% QUARTLE VALUES
        q75 = sub_df.quantile(.75)
        q25=sub_df.quantile(0.25)
        # From this get the interquartile range
        iqr = q75 - q25
        # Calculate the low and high thresholds of IQR for filtering
        min_t = q25 - (iqr*1.5)
        max_t = q75 + (iqr*1.5)
        # Filter result df using thresholds
        sub_df=sub_df[(sub_df>min_t)&(sub_df<max_t)]
        loop_df[y]=sub_df
        loop_df[x]=group_names[i]
        
        # append 
        result_df=result_df.append(loop_df,ignore_index=True)
    # return result df
    return result_df

# runs two-tailed t-tests on groups Not recommended for multiple groups stats analysis
def t_test(var,by,df):
    by_column=df[by]
    by_column=by_column.drop_duplicates()
    p=[]
    comparison_array=[]
    for x in by_column:
        if np.isna(x): continue
        df_x=df[df[by]==x]
        s_x=df_x[var]
        for y in by_column:
            if np.isna(y): continue
            if not x==y:
                if not [x,y] in comparison_array and not [y,x] in comparison_array:
                    df_y=df[df[by]==y]
                    s_y=df_y[var]
                    sig=sp.stats.ttest_ind(s_x,s_y)
                    p.append({'Var':var,'By':by,'X':x,'Y':y,'t':sig[0],'p':sig[1]})
                    comparison_array.append([x,y])
    p=pd.DataFrame(p)
    return p
        

    
    