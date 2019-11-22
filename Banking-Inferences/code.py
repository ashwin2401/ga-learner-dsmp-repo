# --------------
import pandas as pd
import scipy.stats as stats
import math
import numpy as np
import warnings

warnings.filterwarnings('ignore')
#Sample_Size
sample_size=2000

#Z_Critical Score
z_critical = stats.norm.ppf(q = 0.95)  


# path        [File location variable]
data = pd.read_csv(path)
#Code starts here
data_sample = data.sample(n=sample_size, random_state=0)
#print(data_sample)
sample_mean = data_sample.installment.mean()
print(sample_mean)
sample_std = data_sample.installment.std()
print(sample_std)
#finding margin of error
margin_of_error = z_critical*sample_std/math.sqrt(sample_size)
print(margin_of_error)
#finding confidence interval
confidence_interval = (sample_mean-margin_of_error,sample_mean+margin_of_error)
print(confidence_interval)
#storing true mean value of installment column
true_mean = data.installment.mean()
print(true_mean)


# --------------
import matplotlib.pyplot as plt
import numpy as np

#Different sample sizes to take
sample_size=np.array([20,50,100])

#Code starts here
fig,axes = plt.subplots(nrows = 3,ncols= 1)

for i in range (len(sample_size)):
    m = []
    for j in range(1000):
        sampled_data = data['installment'].sample(n=sample_size[i])
        sample_mean = sampled_data.mean()
        m.append(sample_mean)
    mean_series = pd.Series(m)
    axes[i].hist(mean_series)


# --------------
#Importing header files

from statsmodels.stats.weightstats import ztest

#Code starts here
data['int.rate'] = data['int.rate'].str.replace(r'%','')
data['int.rate'] = pd.to_numeric(data['int.rate'])
data['int.rate'] = data['int.rate']/100
x1 = data[data['purpose']=='small_business']['int.rate']
value = data['int.rate'].mean()
z_statistic,p_value = ztest(x1=x1,value=value,alternative='larger')
print('z-statistic =',z_statistic)
print('P-Values =',p_value)
if p_value > 0.05:
    inference = 'Reject'
else:
    inference = 'Accept'

print(inference)


# --------------
#Importing header files
from statsmodels.stats.weightstats import ztest

#Code starts here
z_statistic,p_value = ztest(x1=data[data['paid.back.loan']=='No']['installment'],x2=data[data['paid.back.loan']=='Yes']['installment'])
print('Z-Statistic =',z_statistic)
print('P-Value =',p_value)
if p_value>0.05:
    print('Accept the Null Hypothessis')
else:
    print('Reject the Null Hypothessis')


# --------------
#Importing header files
from scipy.stats import chi2_contingency

#Critical value 
critical_value = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 6)   # Df = number of variable categories(in purpose) - 1

#Code starts here
yes = data[data['paid.back.loan']=='Yes']['purpose'].value_counts()
no = data[data['paid.back.loan']=='No']['purpose'].value_counts()
observed = pd.concat([yes.transpose(),no.transpose()],1,keys=['Yes','No'])

chi2, p, dof, ex = chi2_contingency(observed)

if chi2 > critical_value:
    print('Null Hypothesis is Rejected due to the two distribution are same.  ')
else:
    print('Null Hypothesis cannot be rejected.')


