import pandas as pd
import requests
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
from matplotlib.pyplot import figure

def extract_injections(resp):
    """
    A function which takes a response from FRB master to the injections
    database and returns a time sorted dataframe of injections and
    detections.
    """
    resp_json = resp.json()
    inj = pd.DataFrame(resp_json["injections"]).sort_values("injection_time")
    det = pd.DataFrame(resp_json["detections"]).sort_values("timestamp_utc")
    print("Injections shape: {}".format(inj.shape))
    print("Detections shape: {}".format(det.shape))
    inj["injection_time"] = inj["injection_time"].astype(np.datetime64)
    det["timestamp_utc"] = det["timestamp_utc"].astype(np.datetime64)
	
    data = pd.merge(
        inj,
        det,
        left_on="id",
        right_on="det_id",
        how="outer", #inner will make it so only detected shows up. Outer so that all shows up
        suffixes=("_inj", "_det")
    )
    data = pd.concat(
        [
            data.drop(['extra_injection_parameters'], axis=1),
            data["extra_injection_parameters"].apply(pd.Series)
        ],
        axis=1
    )
    data = data.set_index("timestamp_utc")
    data = data.tz_localize("UTC")
    return data
	
import pandas as pd
import requests
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
from matplotlib.pyplot import figure
import seaborn as sns
sns.set_style("whitegrid")
from matplotlib.pyplot import figure
	
	
def plot_detection_histograms(filtered_dataset, l1_cutoff, l2_cutoff, x_parameter = 'fluence', x_range = None, x_bins = 10, x_log = False):
    """
    Makes a histogram of FRBs detected as a function of x_parameter at bonsai, L1, and L2.
    The filtered dataset is passed as a pandas data frame or a numpy record array and x_parameter must be a valid column name 
    data frame. 
    The x_range can be specified as [x_min, x_max]. 
    If x_range is None, then we use the min and max of the x_parameter in the dataset.
    x_bins specifies the number of bins to split the x axis into.
    """
    
    injections = filtered_dataset
    detections = injections.dropna(how='any')
    l1_pass = detections[detections['avg_l1_rfi_grade']>= l1_cutoff]
    l2_pass = l1_pass[l1_pass['rfi_grade_level2'] >= l2_cutoff]
    
    if x_range is None:
        x_range = [min(injections[x_parameter]),max(injections[x_parameter])]
        
    val = x_range[0]
    n = x_bins-1
    
    if x_log is True:
        x_bins = []
    
        while val<= x_range[1]:
            x_bins.append(val)
            val = x_range[0] + (x_range[1]-x_range[0])/(10**n)
            n= n-1
        
    print("bins:", x_bins)
    hist_injections, bins = np.histogram(injections[x_parameter], bins = x_bins, range=x_range)
    hist_detections, bins = np.histogram(detections[x_parameter], bins = x_bins, range= x_range) 
    hist_l1, bins = np.histogram(l1_pass[x_parameter], bins = x_bins,  range= x_range) 
    hist_l2, bins = np.histogram(l2_pass[x_parameter], bins = x_bins,range= x_range) 
  
    print("bins:", bins) 
    print("Injections Hist:", hist_injections)
    print("Bonsai Hist:", hist_detections)
    print("L1 Hist:", hist_l1) 
    print("L2 Hist:", hist_l2)
    
    fig = plt.figure(figsize =(18, 6))
    
    plt.subplot(1,3,1)
    plt.hist(detections[x_parameter], bins = x_bins, range = x_range, color= 'darkred')
    plt.title("Bonsai")
    if x_log is True:
        plt.xscale('log')
    plt.ylabel("Detections") 
    plt.subplot(1,3,2)
    plt.hist(l1_pass[x_parameter], bins = x_bins, range = x_range, color='darkslategray')
    plt.title("L1")
    if x_log is True:
        plt.xscale('log')
    plt.subplot(1,3,3)
    plt.hist(l2_pass[x_parameter], bins = x_bins, range = x_range)
    plt.title("L2")
    if x_log is True:
        plt.xscale('log')

    return plt.show()
	
sns.set_style("whitegrid")
from matplotlib.pyplot import figure
import pandas as pd
import requests
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
from matplotlib.pyplot import figure

def plot_combined_histograms(filtered_dataset, l1_cutoff, l2_cutoff, x_parameter = 'fluence', x_range = None, x_bins = 10, x_log=False):
    """
    Makes a combined histogram of FRBs detected as a function of x_parameter at bonsai, L1, and L2.
    The filtered dataset is passed as a pandas data frame or a numpy record array and x_parameter must be a valid column name 
    data frame. 
    The x_range can be specified as [x_min, x_max]. 
    If x_range is None, then we use the min and max of the x_parameter in the dataset.
    x_bins specifies the number of bins to split the x axis into.
    """
    
    injections = filtered_dataset
    detections = injections.dropna(how='any')
    l1_pass = detections[detections['avg_l1_rfi_grade']>= l1_cutoff]
    l2_pass = l1_pass[l1_pass['rfi_grade_level2'] >= l2_cutoff]
    
    if x_range is None:
        x_range = [min(injections[x_parameter]),max(injections[x_parameter])]
        
    val = x_range[0]
    n = x_bins-1
    
    if x_log is True:
        x_bins = []
    
        while val<= x_range[1]:
            x_bins.append(val)
            val = x_range[0] + (x_range[1]-x_range[0])/(10**n)
            n= n-1
            
    hist_detections, bins = np.histogram(detections[x_parameter], bins = x_bins)
    hist_l1, bins = np.histogram(l1_pass[x_parameter], bins = x_bins) 
    hist_l2, bins = np.histogram(l2_pass[x_parameter], bins = x_bins) 
    print("bins:", bins)   
    
    fig = plt.figure(figsize =(10, 7))
    plt.hist(detections[x_parameter], bins = x_bins, range = x_range)
    plt.hist(l1_pass[x_parameter], bins = x_bins, range = x_range)
    plt.hist(l2_pass[x_parameter], bins = x_bins, range = x_range)
    plt.title("Numpy Histogram")
    plt.title("Fluence") 
    if x_log is True:
        plt.xscale('log')
    labels= ["Bonsai","L1", "L2"]
    plt.legend(labels)   

    return plt.show()
	
from matplotlib.pyplot import figure
import pandas as pd
import requests
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
from matplotlib.pyplot import figure

from matplotlib.pyplot import figure

def plot_snr_fluence_ratio(filtered_dataset, l1_cutoff, l2_cutoff, x_parameter = 'fluence', x_range = None, x_log=False):
    """
    Makes a detection fraction histogram of FRBs detected as a function of x_parameter at bonsai, L1, and L2.
    The filtered dataset is passed as a pandas data frame or a numpy record array and x_parameter must be a valid column name 
    data frame. 
    The x_range can be specified as [x_min, x_max]. 
    If x_range is None, then we use the min and max of the x_parameter in the dataset.
    """
    injections = filtered_dataset
    detections = injections.dropna(how='any')
    l1_pass = detections[detections['avg_l1_rfi_grade']>= l1_cutoff]
    l2_pass = l1_pass[l1_pass['rfi_grade_level2'] >= l2_cutoff]
    
    plt.figure(figsize=(18,6), dpi=120)
    
    combined_snr_list_bonsai = np.array(detections['combined_snr'].tolist())
    fluence_list_bonsai = np.array(detections['fluence_jy_ms'].tolist())
    combined_snr_list_l1 = np.array(l1_pass['combined_snr'].tolist())
    fluence_list_l1 = np.array(l1_pass['fluence_jy_ms'].tolist())
    combined_snr_list_l2 = np.array(l2_pass['combined_snr'].tolist())
    fluence_list_l2 = np.array(l2_pass['fluence_jy_ms'].tolist())

    snr_fluence_ratio_bonsai = np.divide(combined_snr_list_bonsai,fluence_list_bonsai)
    snr_fluence_ratio_l1 = np.divide(combined_snr_list_l1,fluence_list_l1)
    snr_fluence_ratio_l2 =np.divide(combined_snr_list_l2,fluence_list_l2)

    x_parameter_list_bonsai = np.array(detections[x_parameter].tolist())
    x_parameter_list_l1 = np.array(l1_pass[x_parameter].tolist())
    x_parameter_list_l2 = np.array(l2_pass[x_parameter].tolist())
    
    plt.subplot(1,3,1)
    plt.scatter(x_parameter_list_bonsai, snr_fluence_ratio_bonsai,  c='darkred')
    plt.xlabel(x_parameter)
    if x_log is True:
        plt.xscale('log')
    plt.xlim(x_range)
    plt.title('bonsai')
    plt.subplot(1,3,2)
    plt.scatter(x_parameter_list_l1, snr_fluence_ratio_l1, c = 'midnightblue')
    plt.xlabel(x_parameter)
    if x_log is True:
        plt.xscale('log')
    plt.xlim(x_range)
    plt.title('L1')
    plt.subplot(1,3,3)
    plt.scatter( x_parameter_list_l2, snr_fluence_ratio_l2, c='darkslategray')
    if x_log is True:
        plt.xscale('log')
    plt.xlabel(x_parameter)
    plt.xlim(x_range)
    plt.title('L2')
    
    return plt.show()

def plot_detection_fraction(filtered_dataset, l1_cutoff, l2_cutoff, x_parameter = 'fluence', x_range = None, x_bins = 10, x_log =False ):
    """
    Makes a detection fraction histogram of FRBs detected as a function of x_parameter at bonsai, L1, and L2.
    The filtered dataset is passed as a pandas data frame or a numpy record array and x_parameter must be a valid column name 
    data frame. 
    The x_range can be specified as [x_min, x_max]. 
    If x_range is None, then we use the min and max of the x_parameter in the dataset.
    x_bins specifies the number of bins to split the x axis into.
    """    
    injections = filtered_dataset
    detections = injections.dropna(how='any')
    l1_pass = detections[detections['avg_l1_rfi_grade']>= l1_cutoff]
    l2_pass = l1_pass[l1_pass['rfi_grade_level2'] >= l2_cutoff]
    
    if x_range is None:
        x_range = [min(injections[x_parameter]),max(injections[x_parameter])]
        
    mn = x_range[0]
    mx = x_range[1]
        
    hist_injections, bins = np.histogram(injections[x_parameter], bins = np.logspace(start=np.log10(mn), stop=np.log10(mx), num= x_bins))
    hist_detections, bins = np.histogram(detections[x_parameter], bins = np.logspace(start=np.log10(mn), stop=np.log10(mx), num= x_bins)) 
    hist_l1, bins = np.histogram(l1_pass[x_parameter], bins = np.logspace(start=np.log10(mn), stop=np.log10(mx), num= x_bins)) 
    hist_l2, bins = np.histogram(l2_pass[x_parameter], bins = np.logspace(start=np.log10(mn), stop=np.log10(mx), num= x_bins)) 
    
    print("bins:", bins) 
    print("Injections Hist:", hist_injections)
    print("Bonsai Hist:", hist_detections)
    print("L1 Hist:", hist_l1) 
    print("L2 Hist:", hist_l2)
    
    df_bonsai = np.divide(hist_detections, hist_injections)
    df_l1 = np.divide(hist_l1, hist_injections)
    df_l2 = np.divide(hist_l2, hist_injections)
    
    print("DF at Bonsai:", df_bonsai)
    print('DF at L1:', df_l1)
    print("DF at L2", df_l2)
    
    fig = plt.figure(figsize =(10, 7))
    
    plt.step(bins[1:],df_bonsai)
    plt.step(bins[1:],df_l1)
    plt.step(bins[1:],df_l2)
    plt.ylabel("Detection Fraction")
    plt.xlabel(x_parameter)
    plt.title("Detection fraction as a function of:{}".format(x_parameter)) 
    labels= ["Bonsai","L1", "L2"]
    plt.legend(labels)
    if x_log is True:
        plt.xscale('log')

    return plt.show()