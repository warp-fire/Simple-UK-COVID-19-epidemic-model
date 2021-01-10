# System equations from
# https://www.maa.org/press/periodicals/loci/joma/the-sir-model-for-spread-of-disease-the-differential-equation-modsel
#
# ds_by_dt = -b*s(t)*i(t)            s(0)=1    
# di_by_dt = b*s(t)*i(t) - k*i(t)    i(0)=1.27*10**-6
# dr_by_dt = k*i(t)                  r(0)=0

# Modelling below uses examples from
# https://apmonitor.com/pdc/index.php/Main/SolveDifferentialEquations

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd

	 	
def dZ_by_dt(Z, t, b, k, v):

  s, i, r = Z[0], Z[1], Z[2]

  ds_by_dt = -b * s * i  - v     # (-v = vacination rate added by me)
  di_by_dt = b * s * i - k * i
  dr_by_dt = k * i

  return [ds_by_dt, di_by_dt, dr_by_dt]


key_events = ['sim start', '1st lckdn start', '1st lckdn end', '2nd lckdn start', '2nd lckdn end', 'vac.start', 'xmass start', 'xmass end', '3rd lckdn start', '3rd lckdn end', 'end date']
key_event_dates = ['2020-02-01', '2020-03-23', '2020-08-01', '2020-11-05', '2020-12-02', '2020-12-09', '2020-12-24','2020-12-28', '2021-01-06', '2021-02-28', '2021-12-31'] 

# Create dictionary of event b values under different scenarios
b_dict = {           'Standard':[0.235, 0.046, 0.126, 0.080, 0.130, 0.130, 0.235, 0.130, 0.06, 0.130, 0.130],
		   'Lockdown Christmas':[0.235, 0.046, 0.126, 0.080, 0.130, 0.130, 0.180, 0.130, 0.06, 0.130, 0.130],
                    'Draconian':[0.235, 0.046, 0.110, 0.110, 0.110, 0.110, 0.110, 0.110, 0.06, 0.130, 0.110] }

# Dictionary of vacination rates
vac_dict = {           'Standard':[0, 0, 0, 0, 0, 1/400, 1/400, 1/300, 1/300, 1/300],
             'Lockdown Christmas':[0, 0, 0, 0, 0, 1/400, 1/400, 1/300, 1/300, 1/300],
                      'Draconian':[0, 0, 0, 0, 0, 1/400, 1/400, 1/300, 1/300, 1/300]}


# Loop over dictionaries and run model
s_of_t_dict = {}
i_of_t_dict = {}
r_of_t_dict = {}
	
	
for key in b_dict.keys():
	
	event_b_values = b_dict[key]
	vac_rate = vac_dict[key]
	
	k = event_b_values[0] / 3

	# Initial conditions
	s0 = 1
	i0 = 500 / 66796800 # 500 divided by uk pop from https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/timeseries/ukpop/pop
	r0 = 0
	Z0 = [s0, i0, r0]

	s_of_t, i_of_t, r_of_t = np.empty(0), np.empty(0), np.empty(0)

	# Solve for each timespan with specified b value
	for i in range(len(key_event_dates)-1):

	  dates = np.arange(key_event_dates[i], key_event_dates[i+1], dtype='datetime64[D]')
	  t = np.arange(0, len(dates)+1)
	  
	  # Run the solver
	  Z = odeint(dZ_by_dt, Z0, t, args=((event_b_values[i],k,vac_rate[i])))

	  s_of_t = np.concatenate((s_of_t, Z[:-1, 0]))
	  i_of_t = np.concatenate((i_of_t, Z[:-1, 1]))
	  r_of_t = np.concatenate((r_of_t, Z[:-1, 2]))

	  # Final point for each simulation gives initial conditions for the next
	  Z0 = Z[-1,:]

	
	# Store outputs in dictionary
	s_of_t_dict[key] = s_of_t
	i_of_t_dict[key] = i_of_t
	r_of_t_dict[key] = r_of_t
	
	

# Assumptions for M calc
UK_population = 66796800 # https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/timeseries/ukpop/pop
f_m = .01 # Guessed mortality fraction

# Determine and plot m under each scenario
m_dict = {}

for key in b_dict.keys():
	
	s_of_t = s_of_t_dict[key]
	i_of_t = i_of_t_dict[key]
	r_of_t = r_of_t_dict[key]
		
	m = np.diff(r_of_t) * UK_population * f_m
	# Introduce 14 day lag assuming mortality delay
	m = m[:-14]
	m_dict[key]=m

# Create formatted array of dates	
dates = np.arange(key_event_dates[0], key_event_dates[-1], dtype='datetime64[D]') 
m_dates = dates[:-1] # Determine mortality dates with asssumed 14 day lag
m_dates = m_dates[14:] 

# Plot modelled cases
#plt.figure(0)
plt.subplot(2,1,1)
plt.plot(m_dates,m_dict['Standard'], label='Modelled no Xmas lockdown', color = 'blue')
plt.plot(m_dates,m_dict['Lockdown Christmas'], label='Modelled with Xmas lockdown', color = 'blue', linestyle='dashed')
plt.plot(m_dates,m_dict['Draconian'], label='Modelled with Xmas lockdown', color = 'blue', linestyle='dotted', alpha = 0.2)

# Determine excess m with without Christm#as lockdown
excess_m = int(m_dict['Standard'].sum() - m_dict['Lockdown Christmas'].sum())
print('Excess m is ', excess_m)
	
# Import and plot real M rate against this
df = pd.read_csv('sources/data_2021-Jan-08.csv')
df = df.groupby(['date']).sum()
df = df.add_suffix('_Count').reset_index()
real_m_dates = df['date'].to_numpy(dtype='datetime64[D]')
real_m = df['newDeaths28DaysByDeathDate_Count'].to_numpy()
plt.plot(real_m_dates,real_m,color='red', label='Real')
#plt.xlabel('Date')
plt.ylabel('M')
plt.xticks(rotation=90)
plt.legend()

# plt.subplot(2,2,2)
# real_m_cumulative = np.full_like(real_m,0)
# for i in range(1,len(real_m)):
#     real_m_cumulative[i]=real_m_cumulative[i-1]+real_m[i]
# plt.plot(real_m_dates,real_m_cumulative,color='red', label='Real')
# plt.ylabel('M, cumulative')
# plt.xticks(rotation=90)

# Plot the estimated infection rate
plt.subplot(2,1,2)
plt.plot( dates, i_of_t_dict["Standard"], label='Modelled infection rate', color = 'blue')

# Import and plot real infection rate - recent daily data
df = pd.read_excel('sources/covid19infectionsurveydatasets202101082.xlsx',sheet_name='1b',skiprows=5,engine='openpyxl')
real_I_dates =  df['Unnamed: 0'][:42].to_numpy(dtype='datetime64[D]')
real_I = df['Unnamed: 1'][:42].to_numpy()
plt.plot(real_I_dates,real_I,color='red')

# Import and plot real infection rate - earlier period datas
df = pd.read_excel('sources/datadownload.xlsx',sheet_name='data',skiprows=5,engine='openpyxl')
real_I_earlier_dates =  df['Date'][:19].to_numpy(dtype='datetime64[D]')
real_I_earlier = df['Estimate'][:19].to_numpy()
plt.plot(real_I_earlier_dates,real_I_earlier, label='Real', color='red')

plt.xticks(rotation=90)
#plt.xlabel('Date')
plt.ylabel('I')
plt.legend()


plt.show()