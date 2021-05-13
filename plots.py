import matplotlib.pyplot as plt
import numpy as np
from significance_tests import remove_outliers
import itertools
import seaborn as sns
import pandas as pd

def lite_box_plot(x,y,df,hue='None',order='None',**kwargs):
    remove_outl=True
    
    if kwargs is not None:
        for key,value in kwargs.items():
            if key.lower()=='remove_outl':
                remove_outl=value
            if key.lower()=='order':
                order = value
    # if an order has been passed reorder the dataframe using this
    mean_length=0.5
    if order =='None':
#        df =df.set_index(x)
#        df=df.loc[order]
#        df=df.reset_index()
        group_names=df[x].drop_duplicates()
        group_names=np.array(group_names.values)
    else:
        group_names=order
    # length array for the plot
    length=np.arange(len(group_names))
    # empty arrays to fill in with mean and standard deviation
    mean=np.empty(len(group_names))
    std=np.empty(len(group_names))
    if hue == 'None':
        if remove_outl:
            df=remove_outliers(x,y,df)
        for i in range(len(group_names)):
            
            series=df[df[x]==group_names[i]][y]
            stdev=series.std()
            ave=series.mean()
            std[i]=stdev
            mean[i]=ave
        # plot std using errorbar plot
        plt.errorbar(length,mean,yerr=std,marker='',color='k',capsize=5,ls='None',elinewidth=0.5,markeredgewidth=0.5)
        plt.hlines(mean,length-mean_length/2,length+mean_length/2,linewidth=1/2,color='k')
    else:
        length=length
        sub_cols=df[hue].drop_duplicates()
        sub_cols=sub_cols.values
        sub_cols=sub_cols[::-1]
        spacing=np.linspace(-1/5,1/5,(len(sub_cols)))
        mean_length=mean_length/2
#        spacing=np.zeros(len(hue))
        for i in range(len(sub_cols)):
            sub_df=df[df[hue]==sub_cols[i]]
            if remove_outl:
                sub_df=remove_outliers(x,y,sub_df)
            mean=np.empty(len(group_names))
            std=np.empty(len(group_names))
            for j in range(len(group_names)):
                
                sub_series=sub_df[sub_df[x]==group_names[j]][y]
                stdev=sub_series.std()
                ave=sub_series.mean()
                std[j]=stdev
                mean[j]=ave
            # plot std using errorbar plot
            plt.errorbar(length+spacing[i],mean,yerr=std,marker='',color='k',capsize=5,ls='None',elinewidth=0.5,markeredgewidth=0.5)
            plt.hlines(mean,length-mean_length/2+spacing[i],length+mean_length/2+spacing[i],linewidth=1/2,color='k')
            
            
            

## this dope ass swarm plot mod lets get it boyz   
def better_swarmplot(**kwargs):
    # look through kwargs to see if marker is passes
    if kwargs is not None:
        for key,value in kwargs.items():
            if key.lower()=='marker':
                marker=value
            if key.lower()=='data':
                data=value
    # if a marker variable is declared then we make a plot where the are different markers
    # for all the variables to which the marker was assigned         
    try:marker
    except:
        del data 
        # If if wasn't passed just make a normal swarmplot
        sns.swarmplot(**kwargs)
    else: 
        del kwargs['marker']
        del kwargs['data']
        
        marker_list = itertools.cycle(('^', 'o', 's', 'd')) 
        group_names=data[marker].drop_duplicates()
        group_names=np.array(group_names.values)
        print(group_names)
        count=1
        for g in group_names:
         
            sub_df=data[data[marker]==g]
            sns.swarmplot(marker=next(marker_list),data=sub_df,**kwargs)

def alter_spines(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
   
# code for resampling dataframes by interpolation for plotting
def resample_time_series(x='None',interval=30,df=pd.DataFrame()):
    # select date, condition, and channel number
    id=df['Assay ID'].drop_duplicates()
    id=id.values
    overall_result=pd.DataFrame()
    for i in id:
        df_loop=df[df['Assay ID']==i]
        # determine the min and max of x series then make new  series with interval
        x_series=df_loop[x].values
        x_min=np.min(x_series)
        x_max=np.max(x_series)+interval # go one up because of how arange fun works
        new_x=np.arange(x_min,x_max,interval)
        # declare new df_loop to fill with interpolated values
        result=pd.DataFrame()
        result[x]=new_x
        for column in df_loop:
            if column in ['Max','Mean','Zero Mean','Min']:
                y=column
                y_series=df_loop[y].values
                new_y=np.interp(new_x,x_series,y_series)
                result[y]=new_y
            elif column!='Time (s)':
                y = column
                y_series = df_loop[y].values
                result[y]=[y_series[0]]*len(result)
        overall_result=overall_result.append(result,ignore_index=True)
    return overall_result








        
        
    