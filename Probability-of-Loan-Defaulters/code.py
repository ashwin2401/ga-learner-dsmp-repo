# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataframe
df = pd.read_csv(path)

# probability of  fico score greater than 700

p_a = df[df['fico'].astype(float) >700].shape[0]/df.shape[0]
print(p_a)


# probability of purpose == debt_consolidation
p_b = df[df['purpose']== 'debt_consolidation'].shape[0]/df.shape[0]
print(p_b)

# Create new dataframe for condition ['purpose']== 'debt_consolidation' 
df1 = df[df['purpose']== 'debt_consolidation']

# Calculate the P(A|B)
p_a_b = df1[df1['fico'].astype(float) >700].shape[0]/df1.shape[0]
print(p_a_b)
# Check whether the P(A) and P(B) are independent from each other
result = (p_a == p_a_b)
print(result)


# --------------
# code starts here
prob_lp = (df['paid.back.loan'] == 'Yes').sum() / len(df)
prob_cs = (df['credit.policy'] == 'Yes').sum() / len(df)
new_df = df[df['paid.back.loan'] == 'Yes']
prob_pd_cs = len(new_df[new_df['credit.policy'] == 'Yes']) / len(new_df)
bayes = (prob_pd_cs*prob_lp) / prob_cs
print(bayes)

# code ends here


# --------------
# code starts here
df.purpose.value_counts().plot(kind='bar')
df1 = df[df['paid.back.loan'] == 'No']
df1.plot(kind='bar')

# code ends here


# --------------
# code starts here
inst_median = df.installment.median()
inst_mean = df.installment.mean()
df.installment.hist()
plt.axvline(x=inst_median,color='r')
plt.axvline(x=inst_mean,color='g')
plt.show()
df['log.annual.inc'].hist()
plt.show()
# code ends here


