import pandas as pd
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import skimage.morphology, skimage.data
from scipy.ndimage.morphology import binary_fill_holes
import skimage.filters as f
import skimage.io as io
import math
import os
import javabridge
import bioformats
import vsi_metadata as v
import seaborn as sns
import matplotlib as mpl
import SimpleITK as sitk

def surface_coverage_time_series(filepath,threshEval=False, bw_eval=False,show_all=False,show=False,
                                  highT=27,lowT=20,fill_holes=False,stats=False,show_linear=False,zero_index_time=0,
                                  t_lag_level=.01,zero_index=None,low_bw=1,high_bw=255, edge=True,interval=15,vsi=True,
                                  cycle_vm=True,auto_thresh=True,filter_type='otsu',store_show=False,img_tag='',
                                  meta_number=None,image_channel=1,save_max=False,rescale_factor=255, meta_stage_loop=True,t_sample=1):
    former_path = os.getcwd()
    # if we are reading the data from a .vsi file then use bioformats procedure for reading everything in
    if vsi:
        if cycle_vm:
            javabridge.start_vm(class_path=bioformats.JARS)
        # read in metadata using bioformats and make ararys for t ans z slices
        metadata = v.extract_metadata(filepath, cycle_vm=False,meta_number=meta_number,stage_loop=meta_stage_loop)
        t_slices = np.arange(0, metadata['size_T']-1)
        t_slices=t_slices[::t_sample]
        t_slices_scaled = t_slices * float(metadata['cycle time'])
        z_slices=np.arange(0,metadata['size_Z'])
        scale=metadata['scale']
        # booleans used to turn on switches that show images after 1/3 point on
        # pictures, halfway point and 90% of pictures processed
        not_shown = [True]*3
        area = np.zeros([len(t_slices),2])
        snr=np.zeros(len(t_slices))
        for i in range(len(t_slices)):
            # read in image and convert to 8 bit
            if len(z_slices)==1:
                image = bioformats.load_image(path=filepath, t=t_slices[i], series=0)*rescale_factor
                if len(np.shape(image))>2:
                    image=image[:,:,image_channel]
            else:
                image=max_projection(filepath,t_slices[i],z_slices,image_channel,rescale_factor)
                if save_max:
                    max_proj_folder=former_path+'/Max_proj/'
                    max_proj_folder=max_proj_folder.replace('\\','/')
                    if not os.path.exists(max_proj_folder) :
                        os.mkdir(max_proj_folder)
                    assay_name=os.path.basename(filepath)
                    assay_name=assay_name.replace('.vsi','')
                    assay_folder=max_proj_folder+'/'+assay_name
                    if not os.path.exists(assay_folder) :
                        os.mkdir(assay_folder)
                    img_name=assay_folder+'/'+str(t_slices[i])+'.png'
                    saved_image=(image).astype('uint8')
                    io.imsave(img_name,saved_image)
                ## get max projection
            image = image.astype('uint8')
            mean=np.mean(image)*1E4
            std=np.std(image)*1E4
            snr[i]=mean/std
            show_handler, not_shown,pic_pct = show_controller(i + 1,len(t_slices), *not_shown)
            area[i,:] = calculate_area(image, lowT, highT, i, show_handler, bw_eval, low_bw, high_bw, fill_holes, show,
                  pic_pct, edge, show_all,store_show,img_tag, former_path, auto_thresh,filter_type,scale)
            # print(area[i])
        if cycle_vm:
            javabridge.kill_vm()
    # if we choose not to read in files from a vsi, program autoatically thinks we are loading in a folder full of tifs
    # these must be presorted into the correct fluorescence/ BF channels
    else:
        i = 0
        os.chdir(filepath)
        filenames = sorted(glob.glob('*.tif'))
        pic_length = len(filenames)
        # booleans used to turn on switches that show images after 1/3 point on
        # pictures, halfway point and 90% of pictures processed
        not_shown = [True] * 3
        area = np.zeros([len(filenames),2])
        snr=np.zeros(len(filenames))
        for file in filenames:
            show_handler,not_shown,pic_pct=show_controller(i+1,pic_length,*not_shown)
            # Read in image
            image = cv2.imread(file, 0)
            snr[i]=np.mean(image)/np.std(image)
            # calculate the area of the image
            area[i,:]=calculate_area(image, lowT, highT, i, show_handler, bw_eval, low_bw, high_bw, fill_holes, show,
                  pic_pct, edge, show_all,store_show,img_tag, former_path, auto_thresh,filter_type)
            snr[i]=np.mean(image)/np.std(image)
            i+=1
    # Check to see if a zero index has been if not it then checks to see if a thresh
    if zero_index == None:
        zero_index_thresh = 0.0001
        for i in range(len(area)):
            zero_index = 0
            value = area[i]
            if value > zero_index_thresh:
                zero_index = i
                break
    else:
        zero_index=int(zero_index)
    # reset the surface area array at the zero_index and then subtract the first value so it starts out at zero
    area=area[zero_index:]
    snr=snr[zero_index:]
    zero_area=area-area[0,:]
    # if we're reading from a vsi take the time
    if vsi:
        time=t_slices_scaled[zero_index:]
        time_offset=(time[1]-time[0])*zero_index
        time=time-time_offset
    else:
        assay_time=interval*len(area)
        time=np.linspace(0,assay_time,len(area))
    df=pd.DataFrame({'Time (s)':time,'Surface Coverage':area[:,0],
                     'Zerod Surface Coverage':zero_area[:,0],
                     'Surface Area (um^2)':area[:,1],
                     'Zerod Surface Area (um^2)':zero_area[:,1],
                     'Signal-to-Noise':snr
                     })
    os.chdir(former_path)
    if stats:
        SC_values=get_SC_metrics(df,t_lag_level=t_lag_level,show_linear=show_linear)
        return df,SC_values
    else:
        return df
# controller that tells the return_area function to show a comparison of the thresholded function at set values in the 
# pic thresh list
# controller that tells the return_area function to show a comparison of the thresholded function at set values in the 
# pic thresh list
# controller that tells the return_area function to show a comparison of the thresholded function at set values in the 
# pic thresh list
def show_controller(count_pics,pic_length,*not_shown,pic_thresh=[20,50,90]):
    # cast not shown as list
    not_shown=list(not_shown)
    pic_pct = np.round(count_pics / pic_length * 100, 2)
    if pic_pct >= pic_thresh[0] and not_shown[0] == True:
        show_handler = True
        not_shown[0] = False
    elif pic_pct >= pic_thresh[1] and not_shown[1] == True:
        show_handler = True
        not_shown[1] = False
    elif pic_pct >= pic_thresh[2] and not_shown[2] == True:
        show_handler = True
        not_shown[2] = False
    else:
        show_handler = False
    return show_handler,not_shown,pic_pct

# main workorse of the function, takes in an image, binarizes it, and returns a surface coverage value
# main workorse of the function, takes in an image, binarizes it, and returns a surface coverage value
# main workorse of the function, takes in an image, binarizes it, and returns a surface coverage value
def calculate_area(image, lowT, highT, i, show_handler, bw_eval, low_bw, high_bw, fill_holes, show,
                  pic_pct, edge, show_all,store_show,img_tag, former_path, auto_thresh,filter_type,
                  scale):
    # ------ 1.Threshold the image
    if bw_eval:
        if auto_thresh:
            bw,filt = auto_threshold(image,filter_type,return_filt=True)
        else:
           
            _, bw = cv2.threshold(image, low_bw, high_bw, cv2.THRESH_BINARY)
            filt=low_bw
            
    else:
        bw = image
    # ------ 2.Use edgefinder to identify the bounds of platelet aggregates
    
    if edge:
        # main parts of edgefinding routine
        if auto_thresh:
            edges=auto_canny(bw)
        else:
            edges = cv2.Canny(bw, lowT, highT)
        
        # ----- 3. Fill in the holes in the edgefinder using binary_fill_holes and dilate
        if fill_holes:
            fill = edges
        else:
            # Dilate edges to make unconnected lines connected
            # Kernel to dialiate the image later on and fill the holes. Declared outside of the loop to save on processing power
            kernel = np.ones((3, 3), np.uint8)
            dilate = cv2.dilate(edges, kernel)
            # Fill them holes
            fill = binary_fill_holes(dilate).astype(int)
            # Lines of code I found online to fill in stuff around the edges better
            labels = skimage.morphology.label(fill)
            labelCount = np.bincount(labels.ravel())
            background = np.argmax(labelCount)
            fill[labels != background] = 255
            fill=fill/255
    else:
        # option to bypass edgefinding routine
        fill = bw
    # ----- 4. Determine the surface coverage
    total = float(np.size(fill))
    white = float(np.sum(fill))
    surface_coverage = white / total
    surface_area=white*(scale**2)
    # ------ 5.To ensure that we have binarized the image properly we display the 
    # --------original and binarized image at the 25,50, and 90% progress point of the assay
    #
    # If show is marked we show the final binarized image vs the real image
    # the show_handler regulates when to show
    if show == True and show_handler == True:
        plt.close('all')
        f1=plt.figure(1)
        plt.axis('off')
        plt.suptitle('Assay Percentage {:.1f}'.format(pic_pct))
        plt.subplot(121), plt.imshow(image, cmap='gray');plt.axis('off');
        plt.subplot(122), plt.imshow(fill, cmap='gray');plt.axis('off');
        plt.show()
        f2=plt.figure(2,figsize=(1,1))
        font = {'family' : 'Arial',
        'size'   : 6}
        plt.rc('font', **font)
        sns.distplot(image,kde=False)
        
        try: 
            filt
        except:
            None
        else:
            plt.axvline(x=filt,color='k',linewidth='1',linestyle='--')
        plt.xlim([0,40])
        if store_show:
            storage_path=former_path+'/Analyzed Images/'
            if not os.path.exists(storage_path):
                os.mkdir(storage_path)
            
            norm_string=storage_path+img_tag+'_Original_'+ '{:.1f}'.format(pic_pct)+'.png'
            bin_string=storage_path+img_tag+'_Binarized_'+ '{:.1f}'.format(pic_pct)+'.png'
            hist_string=storage_path+img_tag+'_Histogram_'+ '{:.1f}'.format(pic_pct)+'.svg'
            io.imsave(norm_string,image)
            io.imsave(bin_string,fill)
            f2.savefig(hist_string,dpi=300)
            mpl.rcParams.update(mpl.rcParamsDefault)
    # if show_all is marked True we show the all the images in the steps of the processing
    elif show_all == True and show_handler == True:
        plt.close('all')
        plt.subplot(321), plt.imshow(image, cmap='gray');plt.axis('off');
        plt.title('Original Image')
        plt.subplot(322), plt.imshow(bw, cmap='gray');plt.axis('off');
        plt.title('Thresholded Image')
        plt.subplot(323), plt.imshow(edges, cmap='gray');plt.axis('off');
        plt.title('Edges')
        plt.subplot(324), plt.imshow(dilate, cmap='gray');plt.axis('off');
        plt.title('Dilated')
        plt.subplot(325), plt.imshow(fill, cmap='gray');plt.axis('off');
        plt.title('Filled')
        plt.suptitle('Area: ' + str(surface_coverage) );plt.axis('off');
        plt.subplot(326)
        plt.hist(image.ravel())
        plt.title('Histogram of Original Image')
        plt.show()
    return surface_coverage,surface_area

def max_projection(path,t_slice,z_slices,channel_selection,rescale_factor):
    count=0
    for z in z_slices:
        if z==np.min(z_slices):
            test_img = bioformats.load_image(path=path, t=t_slice,z=z, series=0)
            shape=np.shape(test_img)
            img_array=np.zeros([shape[0],shape[1],len(z_slices)])
        img_loop=bioformats.load_image(path=path, t=t_slice,z=z)*rescale_factor
        if len(np.shape(img_loop))>2:
            img_loop=img_loop[:,:,channel_selection]
        img_array[:,:,count]=img_loop
        count+=1
    max_intensity=np.max(img_array,axis=2)
    return max_intensity
            
    

def auto_threshold(image,filter_type,return_filt=False):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    if filter_type=='otsu':
        filt= f.threshold_otsu(blurred) 
        bw= blurred>filt
        bw=bw.astype(int)
    elif filter_type=='isodata':
        filt= f.threshold_isodata(blurred) 
        bw= blurred>filt
        bw=bw.astype(int)
    elif filter_type=='triangle':
        filt= f.threshold_triangle(blurred) 
        bw= blurred>filt
        bw=bw.astype(int)
    elif filter_type=='entropy':
        filt= f.threshold_li(blurred) 
        bw= blurred>filt
        bw=bw.astype(int)
    elif filter_type=='mixed':
        snr=np.mean(blurred)/np.std(blurred)
        if snr>2:
             filt= f.threshold_triangle(blurred) 
        else:
            filt= f.threshold_otsu(blurred) 
        bw= image>filt
        bw=bw.astype(int)     
    if return_filt:
        return bw,filt
    else:
        return bw

def auto_canny(image, sigma=0.99,std_eval=True,correct_fact=1):
    image=np.float32(image)
    blurred = cv2.GaussianBlur(image, (3, 3),0)
    blurred= (blurred).astype('uint8')
    #compute the median of the single channel pixel intensities
    v = np.mean(blurred)
    std=np.std(blurred)
    mean_calc=v
    std_calc=std
    snr=mean_calc/std_calc
    # if snr>3:
    #     correct_fact=1
	# apply automatic Canny edge detection using the computed median
    if std_eval:
        lower = int((v-2*std)*correct_fact)
        upper = int((v+2*std)*correct_fact*correct_fact)
    else:
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(blurred, lower, upper)
    return edged 


def get_SC_metrics(df,show_linear=False,max_t_lag=5*60,t_lag_level=0.01,interval=10,
                   y_var='Surface Area (um^2)'):
    # get min and max surface coverages
    maxSC = np.max(df[y_var])
    minSC = np.min(df[y_var]) + 0.000001
    # resample series to finer grain for better analysis 
    x,y=interpolate_series(df,y_var=y_var)
    # get index of 80% val of max
    max_index=get_max_index(x,y)
    # get lag time and update certain indices if need be
    t_lag,t_lag_index,max_index=get_lag_time(x,y,t_lag_level,max_t_lag,max_index,maxSC)
    # Get slope of linear region
    slope,intercept=get_slope(x,y,t_lag_index,max_index)
    if show_linear:
        # Plot raw data 
        plt.figure(1)
        y_linear = x[t_lag_index:max_index] * slope + intercept
        plt.plot(x[t_lag_index:max_index], y_linear, '-r', label='Linear Fit', linewidth=2)
        plt.plot(df['Time (s)'], df['Surface Area (um^2)'], 'ob', label='Experimental Data',alpha=0.5)
        plt.axvline(x=t_lag, color='k', linestyle='--', label='T-lag')
        plt.axhline(y=maxSC, color='g', linestyle='--', label='Max')
        plt.xlabel('Time (s)')
        plt.ylabel('Surface Area - um^2')
        plt.title('Experimental Data and Fitted Kinetic Metrics')
        plt.legend()
        plt.show()
    # Store data using pandas 
    SC_values = pd.DataFrame([{'SA T-lag': t_lag, 'SA Max': maxSC, 'SA Slope': slope}])
    return SC_values

def get_slope(x,y,t_lag_index,max_index,metric='max_slope',min_fit=0.5):
    min_fit_length = int(round(min_fit*(len(x)+1)))
    if max_index>=100:
        iterator=int(round(len(x)/100))
    else:
        iterator=1
    # Now the code chooses the slope on the criteria of either:
    # - getting the largest possible slope of a line that is min_fit*the fitting length
    if metric=='max_slope':
        coefs=max_slope(x,y,t_lag_index,max_index,iterator,min_fit_length)
    # - or by getting the most linear region for the fitting space
    if metric=='best_fit':
        # Search on the interval from t-lag to 2% from the max value
        coefs=chi_squared_min(x,y,t_lag_index,max_index,iterator,min_fit_length)
        
    slope=coefs[0]
    # no negative slopes!
    if slope<0: slope =0
    try: 
        coefs[1]
    except: 
        intercept=0
    else:
        intercept=coefs[1]
    
    return slope,intercept

def max_slope(x,y,t_lag_index,max_index,iterator,min_fit_length):
    slope_best=0
    i_best=t_lag_index
    j_best=max_index
    j_best=max_index
    for i in range(t_lag_index, max_index ,iterator):
        for j in range(i+min_fit_length, max_index,iterator):
            # If the array is zero 
            if not x[i:j].size or not y[i:j].size:
                print ('Array Empty in Loop')
                print (i,j)
            elif not np.absolute(j-i)<min_fit_length:
                coefs_loop = np.polyfit(x[i:j], y[i:j], 1)
                slope_loop=coefs_loop[0]
                if slope_loop>slope_best:
                    i_best = i
                    j_best = j
                    slope_best=slope_loop
    if not x[i_best:j_best].size or not y[i_best:j_best].size:
        print('Array Empty Outside of Loop')
        print(i_best,j_best)
        coefs=[float('nan')]
    else:
          coefs = np.polyfit(x[i_best:j_best], y[i_best:j_best], 1)
    return coefs

def chi_squared_min(x,y,t_lag_index,max_index,iterator,min_fit_length):
    chi_min=1E-3
    for i in range(t_lag_index, max_index ,iterator):
        for j in range(i+min_fit_length, max_index,iterator):
            # If the array is zero 
            if not x[i:j].size or not y[i:j].size:
                print ('Array Empty in Loop')
                print (i,j)
            elif not np.absolute(j-i)<min_fit_length:
                coefs_loop = np.polyfit(x[i:j], y[i:j], 1)
                y_linear = x * coefs_loop[0] + coefs_loop[1]
                chi = 0
                for k in range(i, j):
                    chi += (y_linear[k] - y[k]) ** 2

                if chi < chi_min:
                    i_best = i
                    j_best = j
                    chi_min = chi
                # print 'Chi-min: '+str(chi_min)
                # print 'Chi:'+str(chi)
    if not x[i_best:j_best].size or not y[i_best:j_best].size:
        print('Array Empty Outside of Loop')
        print(i_best,j_best)
        coefs=[float('nan')]
    else:
        coefs = np.polyfit(x[i_best:j_best], y[i_best:j_best], 1)
    return coefs

# Get index for first data point 2% away from the max
def get_lag_time(x,y,t_lag_level,max_t_lag,max_index,maxSC):
    y=y-y[0]
    t_lag_index=None
    for i in range(len(y)):
        # If the surface coverage is greater or equal 5% then the time at that point is stored and the loop ends
        if round(y[i], 2) >= t_lag_level:
            t_lag = x[i]
            t_lag_index = i
            break
    # if a lag time wasn't picked up auto assing the maximum
    if t_lag_index == None:
        t_lag=max_t_lag
        t_lag_index=0
    # If the lag time is huge, change the index we pass the slope fitter
    if t_lag_index>0.6*len(x):
        t_lag_index=0

    return t_lag,t_lag_index,max_index

# Get index for first data point 20% away from the max
def get_max_index(x,y,tolerance=0.2):
    max_index=-1
    max_SC=np.max(y)
    for i in range(len(y)):
        difference = np.absolute((y[i] - max_SC) / max_SC)
        if difference <= tolerance:
            max_index = i
            break
    if max_SC<100:
        max_index=len(y)+1
    return max_index

def interpolate_series(df,interp_interval=1,y_var='Surface Area (um^2)',
                       x_var='Time (s)'):
    # Get index
    interval=df[x_var][1]-df[x_var][0]
    start=np.min(df[x_var])
    stop=np.max(df[x_var])+interval
    x=np.arange(start,stop,1)
    y=np.interp(x,df[x_var],df[y_var])
    return x,y