#!/usr/bin/env python
# coding: utf-8

# # Introduction

# We work at a startup that sells food products. Our mission is to investigate user behavior for the company's app.
# 
# First we'll study the sales funnel. Find out how users reach the purchase stage. How many users actually make it to this stage? How many get stuck at previous stages? Which stages in particular?
# 
# Then we'll look at the results of an A/A/B test. The designers would like to change the fonts for the entire app, but the managers are afraid the users might find the new design intimidating. They decide to make a decision based on the results of an A/A/B test.
# 
# The users are split into three groups: two control groups get the old fonts and one test group gets the new ones. We'll find out which set of fonts produces better results.
# 
# Creating two A groups has certain advantages. We can make it a principle that we will only be confident in the accuracy of our testing when the two control groups are similar. If there are significant differences between the A groups, this can help us uncover factors that may be distorting the results. Comparing control groups also tells us how much time and data we'll need when running further tests.
# 
# We'll be using the same dataset for general analytics and for A/A/B analysis.

# **Description of the data:**
# 
# Each log entry is a user action or an event.
# - EventName — event name
# - DeviceIDHash — unique user identifier
# - EventTimestamp — event time
# - ExpId — experiment number: 246 and 247 are the control groups, 248 is the test group

# <h1>Tables Of Contents <a class="anchor" id="table_of_contents"></a></h1>

# - [1. Open the data file and read the general information](#open)
# - [2. Prepare the data for analysis](#prepare)
# - [3. Study and check the data](#check)
# - [4. Study the event funnel](#funnel)
# - [5. Study the results of the experiment](#result)
# - [6. Overall conclusion](#conclusion)

# # Open the data file and read the general information

# <a class="anchor" id="open"></a>
# [Go back to the Table of Contents](#table_of_contents)

# Install libraries:

# In[1]:


# !pip install -Uq plotly
# !pip install sidetable -U
# !pip install -U seaborn


# Import necessary libraries:

# In[2]:


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly import graph_objects as go
import sidetable
from scipy import stats as st
from datetime import datetime
import sys
import warnings


# Load the data file to variable `data`:

# In[3]:


try:
    data = pd.read_csv('logs_exp_us.csv', sep='\t')
except:
    data = pd.read_csv('/datasets/logs_exp_us.csv', sep='\t')


# In[4]:


# ignore 'FutureWarning'
if not sys.warnoptions:
       warnings.simplefilter("ignore")


# Let's take a look at the general information of the data:

# In[5]:


# print first 5 rows of data
data.head()


# As mentioned before, values in `EventTimestamp` should be converted to the proper timestamp format. Besides that, all the other columns have the correct data types. 
# 
# However, we should check how many unique values there are in `EventName`, perhaps there are only a select few and we can convert the column's data type to `category` to save memory space.

# Using `info()` method we'll check the data types of each column:

# In[6]:


data.info()


# As mentioned before, values in `EventTimestamp` should be converted to the proper timestamp format. Besides that, all the other columns have the correct data types.
# 
# However, we should check how many unique values there are in `EventName`, perhaps there are only select few and we can convert the column's data type to `category` to save memory space.

# ## Conclusion

# To conclude, we loaded an event tracking data file to the variable `data`.
# 
# From the general look at the data, we found the following issues:
# - Columns names are not clear and contain uppercase letters, which is a bad practice.
# 
# 
# - Values in `EventTimestamp` have the wrong format and data type.
# 
# We'll address these issues during the data preprocessing stage.

# # Prepare the data for analysis

# <a class="anchor" id="prepare"></a>
# [Go back to the Table of Contents](#table_of_contents)

# Let's fix the issues we discussed before, we'll change the columns names to comply with python programming conventions:

# In[7]:


data.columns = ['event_name', 'user_id', 'event_timestamp', 'exp_id']


# Using `datetime.fromtimestamp()` method we'll convert the values in `event_timestamp` to the proper format:

# In[8]:


data['event_timestamp'] = data['event_timestamp'].apply(lambda x:datetime.fromtimestamp(x))


# Now, we'll add a date column to `data` (the date an event has occurred):

# In[9]:


# extract date from 'event_timestamp' column
data['date'] = data['event_timestamp'].dt.date

# convert the new column to datetime data type
data['date'] = data['date'].astype('datetime64')


# Next, using the `stb.missing()` method we'll search for missing values in `data` and their share out of the total values:

# In[10]:


data.stb.missing(style=True)


# No missing values found!

# We didn't find any missing values, but that doesn't mean the data is clean.
# 
# Using the `describe()` method we'll look at the basic statistical information of `data` and search for anomalies/outliers:

# In[11]:


data.describe(include='all')


# General key points from `data`:
# - The test duration was from 2019-07-25 to 2019-08-08.
# - 'MainScreenAppear' is the most frequent event in `data` with 119,101 instances.
# - '2019-08-06' is the most frequent date in `data` with 36,270 instances.

# Only 5 unique values are in `event_name`, we can change the column's data type to `category` in order to save memory space.

# In[12]:


# using astype() method to change data type of 'event_name'
data['event_name'] = data['event_name'].astype('category')


# Other than that the data looks fine, let's check if there are any duplicates in the data:

# In[13]:


print('The number of duplicates in the data is: {}'.format(data.duplicated().sum()))


# 413 duplicates found, these duplicates accounts for less than 1% of the data, so deleting them won't impact our analysis significantly.

# In[14]:


# drop the duplicates we found earlier
data = data.drop_duplicates().reset_index(drop='inplace')


# Finally, the data preprocessing stage is complete!

# ## Conclusion

# To conclude, at this step we addressed the issues we mentioned earlier:
# 1. We renamed the columns, made the names clearer, and replaced uppercase letters with lowercase letters, as expected from a python programmer.
# 
# 
# 2. We converted the `event_timestamp` column's values to a date and time format. Moreover, We added a `date` column to display the dates of the logs without time indicators.
# 
# We didn't find any missing values in `data`.
# 
# Next, we used the `describe()` method to get basic statistical information about the data, here are general interest points:
# 
# - The test duration was from 2019-07-25 to 2019-08-08.
# 
# 
# - 'MainScreenAppear' is the most frequent event in `data` with 119,101 instances.
# 
# 
# - '2019-08-06' is the most frequent date in `data` with 36,270 instances.
# 
# Because we found only five unique values in `event_name`. We changed the column's data type to `category` to save memory space.
# 
# Eventually, we searched for duplicates in `data`, we found 413 duplicates.
# These duplicates accounted for less than 1% of the data, so deleting them didn't impact our analysis significantly.

# # Study and check the data

# <a class="anchor" id="check"></a>
# [Go back to the Table of Contents](#table_of_contents)

# This stage will involve answering a few general questions and digging for insights.

# - **How many events are in the logs?**

# Let's count how many events there are in `data`, and print it: 

# In[15]:


print('The number of events in the logs is: {}'.format(len(data)))


# - **How many users are in the logs?**

# Let's calculate how many unique users there are in `data`:

# In[16]:


print('The number of users in the logs is: {}'.format(data['user_id'].nunique()))


# - **What's the average number of events per user?**

# Let's calculate the average number of events per user, and print it:

# In[17]:


print('The average number of events per users is: {:.2f}'.format(data.groupby('user_id')['event_name'].count().mean()))


# - **What period of time does the data cover?**

# Earlier in the project we used the `describe()` method and we found that the test duration was from 2019-07-25 to 2019-08-08.

# Now, let's plot a histogram to examine the distribution of events by date and time:

# In[18]:


# plot a histogram using plotly express library
fig = px.histogram(data, x='event_timestamp')

# set title and x/y labels
fig.update_layout(
    title='Distribution of Events by Date and Time',
    xaxis_title='Date and Time',
    yaxis_title='Frequency')
fig.show()


# From the histogram above we understand that the data before August 1 is incomplete. Therefore, we'll filter the data and use only data from August 1, 2019.
# 
# A possible explanation for the incomplete data is that somehow a portion of the users actions were mistakenly added to the dataset before the test began.

# In[19]:


# using the query() method we'll filter the original dataset 
# and take events that occurred from Aug 1 till the end of the test
filtered_data = data.query('date >= "2019-08-01"')


# Now the data we have represent the period from 2019-08-01 to 2019-08-08, this is the relevant period for our test.

# - **Did we lose many events and users when excluding the older data?**

# Let's calculate how many events and users we lost after the filtering of the original data:

# In[20]:


# the number of events in the original dataframe 
events_original_len = len(data)
# the number of events in the filtered dataframe
events_filtered_len = len(filtered_data)

# calculate the difference between the variables above
lost_events = events_original_len - events_filtered_len

# the number of unique users in the original dataframe 
users_original_len = data['user_id'].nunique()
# the number of unique users in the filtered dataframe
users_filtered_len = filtered_data['user_id'].nunique()

# calculate the difference between the variables above
lost_users = users_original_len - users_filtered_len

# print the results
print('The number of events lost is {} which account for {:.2%} of the original data'
      .format(lost_events, (lost_events/events_original_len)))
print()
print('The number of users lost is {} which account for {:.2%} of the original data'
      .format(lost_users, (lost_users/data['user_id'].nunique())))


# As you can see above, the lost information numbers are insignificant and the majority of the logs from the original dataframe remains.

# - **Make sure we have users from all three experimental groups.**

# In[21]:


# find the number of unique users in each group
filtered_data.groupby('exp_id')['user_id'].agg('nunique')


# As you can see above, we have users from all three experimental groups.

# ## Conclusion

# To sum up, at this stage we studied the data, here are the main points of interest we were able to draw:
# 
# - The number of events in the logs is 243,713.
# 
# 
# - The average number of events per user is 32.28.
# 
# 
# - The test duration was from 2019-07-25 to 2019-08-08, but data before August 1 was incomplete. Therefore, we filtered the data and used only data from August 1, 2019.
# 
# 
# - The number of events lost after the filtering is 1989, which accounted for 0.82% of the original data.
# 
# 
# 
# - The number of users lost after the filtering is 13, which accounted for 0.17% of the original data.
# 
# 
# - We found 2,484 unique users for group 246 (first control group), 2,517 unique users for 247 (second control group), and 2,537 unique users for group 248 (test group).

# # Study the event funnel

# <a class="anchor" id="funnel"></a>
# [Go back to the Table of Contents](#table_of_contents)

# Taking a deeper look at the event funnel will provide useful insights:

# **See what events are in the logs and their frequency of occurrence, and sort them by frequency.**

# Let's find what events are in the logs and their frequency of occurrence, we'll print them sorted in a descending order:

# In[22]:


events_freq = filtered_data.groupby('event_name', as_index=False)['user_id'].count().sort_values(
    by='user_id', ascending=False)
events_freq.columns = ['event_name', 'count']


# Let's visualize this with barplot:

# In[23]:


# set the plot's style
sns.set_style('dark')

# set the figure size
plt.figure(figsize=(12, 8))

ax = sns.barplot(x='event_name', y='count', data=events_freq, order=events_freq['event_name'])

# add title and x/y labels
plt.title('Events types and their frequencies', size=18)
plt.xlabel('Event Name', size=14)
plt.ylabel('Frequency', size=14)

plt.show()


# Overall the results make sense, 'MainScreenAppear' event occur the most since it's the landing page where all users start their 'journy' in our website, 'Toturial' isn't mandatory and most users don't use it.

# **Find the number of users who performed each of these actions. Sort the events by the number of users. Calculate the proportion of users who performed the action at least once.**

# In[24]:


# calculate the amount of users who performed each of the events and sort them in descending order
n_users_by_events = filtered_data.groupby('event_name', as_index=False)['user_id'].nunique().sort_values(
    by='user_id', ascending=False)
n_users_by_events.columns = ['event_name', 'n_users']

# add a share column for the table we made above, calculate the share of users who performed an action out of all users
n_users_by_events['share'] = (n_users_by_events['n_users'] / filtered_data['user_id'].nunique()) * 100

# print the result
n_users_by_events


# Here are the numbers we looked for, almost 47% of users bought something from our website, that's a good conversion rate.

# **In what order the actions took place? Are all of them part of a single sequence?**

# The logical order of actions would be:
# 1. MainScreenAppear
# 2. OfferScreenAppear
# 3. CartScreenAppear
# 4. PaymentScreenSuccessful
# 
# We didn't include 'Tutorial' because only 11% of users took this action, that suggest this action isn't part of the funnel. So we'll exclude 'Tutorial' from now on as it's not mandatory for our main goal (Users buying products).

# Now, we can't trust our common sense alone, we'll search for the most 'popular' sequence of events.

# Let's sort our data (excluding 'Tutorial') by `user_id` and `event_timestamp`: 

# In[25]:


sequence_data = filtered_data[filtered_data['event_name'] != 'Tutorial'].sort_values(by=['user_id','event_timestamp'])
# print first 5 rows
sequence_data.head()


# Next, we'll write a function that will return the sequence of actions for a given `user_id`:

# In[26]:


def sequence(user_id):
    # find the the sequence of actions sorted by time of action for the user_id given in the parameter
    # and save it to a new df
    sequence_of_actions = sequence_data[sequence_data['user_id'] == user_id].sort_values(by=['user_id','event_timestamp'])
    # return the sequence of actions as a list without duplicates 
    return sequence_of_actions['event_name'].drop_duplicates().to_list()


# Now, we'll use the function we wrote to make a list of unique users and their first sequence of actions in our website:

# In[27]:


# create an empty list  
sequence_list = []

# loop through sequence_data df we made earlier and for every unique user, use the function 'sequence'
# to find the user's first sequence of actions, add every sequence to the list 
for i in sequence_data['user_id'].unique():
    sequence_list.append([i,sequence(i)])


# Next, we'll create a dataframe from the list we made above:

# In[28]:


path_data = pd.DataFrame(sequence_list, columns = ['user','sequence'])

# convert sequence column to 'str' data type instead of a list
path_data['sequence'] = path_data['sequence'].astype('str')
# print first 5 rows
path_data.head()


# Using `value_counts()` method, we'll find the most 'popular' sequences of events:

# In[29]:


path_data['sequence'].value_counts()


# Not all sequences contain all of the actions, but as we thought the sequence:
# 1. MainScreenAppear
# 2. OffersScreenAppear
# 3. CartScreenAppear
# 4. PaymentScreenSuccessful
# 
# Is the most 'popular' sequence of actions (with all the actions taken) with 912 instances.

# **Use the event funnel to find the share of users that proceed from each stage to the next. (For instance, for the sequence of events A → B → C, calculate the ratio of users at stage B to the number of users at stage A and the ratio of users at stage C to the number at stage B.)**

# First, we'll sort group `filtered_data` by `event_name` and count how many users reached each event (excluding 'Tutorial'), and sort the table by the number of users at each event.

# In[30]:


event_funnel = filtered_data[filtered_data['event_name'] != 'Tutorial'].groupby(['event_name'])['user_id'].nunique().sort_values(
    ascending=False).reset_index()
# drop the row of 'Tutorial' event
event_funnel.drop(labels= 4, axis=0, inplace=True)
event_funnel


# Next, we'll calculate the share of users lost at each step (each event):

# In[31]:


# add a percent change column to event_funnel df using the pct_change() method
event_funnel['percent_change'] = event_funnel['user_id'].pct_change()
event_funnel


# Above we can see the share of users who are lost from an event to the next event, but to get a meaningful information we need to split the funnel between the test groups.
# 
# We'll create an empty list, and loop through `filtere_data`. For each  unique `exp_id` we'll find the event funnel similar to the way we did it earlier.

# In[32]:


# create an empty list
funnel_by_groups = []

# loop through the data for each unique test group, find how many unique users reached each event
# and sort the information by the number of users
for i in filtered_data.exp_id.unique():
    group = filtered_data[filtered_data.exp_id == i].groupby(
        ['event_name','exp_id'])['user_id'].nunique().reset_index().sort_values(
        by='user_id',ascending=False)
    
    # drop the rows with 'Tutorial' in event_name column
    group.drop(labels= 4, axis=0, inplace=True)
    # display temporary list for number i test group
    display(group)
    # add all the information we gathered to a single list
    funnel_by_groups.append(group)


# Now, we'll concatenate the list we made above and create a dataframe out of it in order to visualize the results later. 

# In[33]:


funnel_by_groups = pd.concat(funnel_by_groups)
funnel_by_groups


# Let's visualize the event funnel for each test group:

# In[34]:


fig = go.Figure()

# funnel for experiment group number 246
fig.add_trace(go.Funnel(
    name = 'Experiment Group - 246',
    y = list(funnel_by_groups[funnel_by_groups['exp_id'] == 246]['event_name']),
    x = list(funnel_by_groups[funnel_by_groups['exp_id'] == 246]['user_id']),
    textinfo = "value+percent initial"))

# funnel for experiment group number 247
fig.add_trace(go.Funnel(
    name = 'Experiment Group - 247',
    y = list(funnel_by_groups[funnel_by_groups['exp_id'] == 247]['event_name']),
    x = list(funnel_by_groups[funnel_by_groups['exp_id'] == 247]['user_id']),
    textinfo = "value+percent initial"))

# funnel for experiment group number 248
fig.add_trace(go.Funnel(
    name = 'Experiment Group - 248',
    y = list(funnel_by_groups[funnel_by_groups['exp_id'] == 248]['event_name']),
    x = list(funnel_by_groups[funnel_by_groups['exp_id'] == 248]['user_id']),
    textinfo = "value+percent initial"))

# set title and labels
fig.update_layout(
    title="Event Funnel by Experiment Group",
    yaxis_title="Number and Share of Users",
    legend_title="Experiment Group")

fig.show()


# From the initial look at the funnel for each experiiment group, we can say that the differences in conversion between groups are pretty small. We'll test for significant difference later in the project.

# **At what stage do you lose the most users?**

# For all experiment groups, the largest chunk of users are lost at the 'OffersScreenAppear', almost 40% of users in all groups are lost by the time they reach this event.

# **What share of users make the entire journey from their first event to payment?** 

# - For group number 246: 49% of users reach 'PaymentScreenSuccessful' event.
# 
# 
# - For group number 247: 47% of users reach 'PaymentScreenSuccessful' event.
# 
# 
# - For group number 248: 47% of users reach 'PaymentScreenSuccessful' event.
# 
# Overall, across all groups, that's a good conversion rate.
# 
# With that being said there is room for improvement. We lose almost 40% of users at the 'OffersScreenAppear' event.
# 
# Making the offers in 'OffersScreenAppear' more attractive and irresistible to users could lead to a higher conversion rate at this stage, therefore leading to higher conversion to purchases (our main goal). 

# ## Conclusion

# To conclude, at this stage we studied the event funnel, we found the following points of interest:
# 
# - We found five different events are in the logs and their frequency of occurrence. Also, we found the number of users who performed each action and their share out of total users. Here they are sorted in descending order:
# 
#     1. 'MainScreenAppear', 7,423 users, 98.45% share.
#     2. 'OfferScreenAppear', 4,597 users, 60.98% share.
#     3. 'CartScreenAppear', 3,736 users, 49.56% share.
#     4. 'PaymentScreenSuccessful', 3,540 users, 46.96% share.
#     5. 'Tutorial', 843 users, 11.18% share.
# 
# 
# - Afterwards, we excluded 'Tutorial' event from the rest of project, because it's irrelevant for the event funnel.
# 
# 
# - Not all sequences of actions contain all of the actions, but the most 'popular' sequence of actions (with all the actions taken) with 912 instances is:
# 
#     1. MainScreenAppear
#     2. OffersScreenAppear
#     3. CartScreenAppear
#     4. PaymentScreenSuccessful
# 
# 
# - For all experiment groups, the largest chunk of users are lost at the 'OffersScreenAppear', almost 40% of users in all groups are lost by the time they reach this event.
# 
# 
# 
# -  Share of users make the entire journey from their first event to payment:
#     - For group number 246: 49% of users reach 'PaymentScreenSuccessful' event.
# 
#     - For group number 247: 47% of users reach 'PaymentScreenSuccessful' event.
# 
#     - For group number 248: 47% of users reach 'PaymentScreenSuccessful' event.
#     
# Making the offers in 'OffersScreenAppear' more attractive and irresistible to users could lead to a higher conversion rate at this stage, therefore leading to higher conversion to purchases.

# # Study the results of the experiment

# <a class="anchor" id="result"></a>
# [Go back to the Table of Contents](#table_of_contents)

# **How many users are there in each group?**

# In[35]:


# create a dictionary with the number of unique users in each experiment group
users_by_group = dict(filtered_data.groupby(['exp_id'])['user_id'].nunique())
# print the results
for i in users_by_group:
    print('The number of users in experiment group {} is: {}. Which account for {:.2%} of the total number of users'.format(
        i, users_by_group[i], users_by_group[i] / filtered_data['user_id'].nunique()))
    print()


# Seems like the groups were split relatively evenly.

# Let's validate that each user was assigned to one group only:

# In[36]:


# for each user, check how many unique groups was he assigned to
# print the number of users who was assigned to more than one group
len(filtered_data.groupby(['user_id'])['exp_id'].nunique().reset_index().query('exp_id > 1'))


# Great! All the users were assigned to one group only.

# **We have two control groups in the A/A test, where we check our mechanisms and calculations. See if there is a statistically significant difference between samples 246 and 247:**
# 
# - Select the most popular event. In each of the control groups, find the number of users who performed this action. Find their share. Check whether the difference between the groups is statistically significant. Repeat the procedure for all other events (it will save time if you create a special function for this test). Can you confirm that the groups were split properly?

# At previous stages, we saw that 'MainScreenAppear' is the most popular event across al groups.
# 
# Let's create a pivot table that aggregate the number of users that reached each event for each group:

# In[37]:


pivot = filtered_data.pivot_table(
    index='event_name', columns='exp_id', values='user_id', aggfunc = lambda x: x.nunique()).reset_index()

# sort the pivot table by the most popular events, drop the 'Tutorial' event since it's irrelevant for the funnel
pivot = pivot.sort_values(by= [246,247,248] , ascending=False).drop(labels= 4, axis=0).reset_index(drop=True)
# print pivot table
pivot


# Here we can see clearly, 'MainScreenAppear' is the most popular event across al groups.
# 
# Let's write a function to check if there is a statistically significant difference between samples proportions:

# In[38]:


def check_hypothesis(group1, group2, event, alpha):
    
    # find the number of successes for each group
    success1 = pivot[pivot.event_name == event][group1].iloc[0]
    success2 = pivot[pivot.event_name == event][group2].iloc[0]
    
    # find the number of trials for each group
    trials1 = filtered_data[filtered_data.exp_id == group1]['user_id'].nunique()
    # check if trials1 equal 0, then set trials1 to be total unique users in 246 & 247 combined
    if trials1 == 0:
        trials1 = filtered_data[(filtered_data.exp_id == 246) | (filtered_data.exp_id == 247)]['user_id'].nunique()
    trials2 = filtered_data[filtered_data.exp_id == group2]['user_id'].nunique()
    
    
    # success proportion in the first group:
    p1 = success1 / trials1

    # success proportion in the second group:
    p2 = success2 / trials2

    # success proportion in the combined dataset:
    p_combined = (success1 + success2) / (trials1 + trials2)

    # the difference between the datasets' proportions
    difference = p1 - p2
    # calculating the statistic in standard deviations of the standard normal distribution
    z_value = difference / math.sqrt(p_combined * (1 - p_combined) * (1 / trials1 + 1 / trials2))

    # setting up the standard normal distribution (mean 0, standard deviation 1)
    distr = st.norm(0, 1)
    # calculating the statistic in standard deviations of the standard normal distribution


    p_value = (1 - distr.cdf(abs(z_value))) * 2

    print('p-value: ', p_value)

    if (p_value < alpha):
        print("Rejecting the null hypothesis for", event, "and groups", group1, group2)
    else:
        print("Failed to reject the null hypothesis for", event, "and groups", group1, group2)


# Before we apply our function to test the data, let's formulate our hypotheses.
# 
# For a given two sample groups, and a certain event:
# 
# - **The null hypothesis is:** For group A and group B, the shares of users who reached a certain event out of the total users at each group are the same.
# 
# 
# - **The alternative hypothesis is:** For group A and group B, the shares of users who reached a certain event out of the total users at each group are not the same.
# 
# We'll use significance criterion (alpha) of 5% since it's the golden standard for the industry.

# Now, let's apply the function for our control groups: 246, 247, and the most popular event: 'MainScreenAppear'.

# In[39]:


check_hypothesis(246, 247, 'MainScreenAppear', 0.05)


# We failed to reject the null hypothesis, let's apply the same function for the other events:

# In[40]:


# loop through all the events and apply the test function except 'MainScreenAppear' that we already tested
for i in pivot[pivot['event_name'] != 'MainScreenAppear'].event_name.unique():
    check_hypothesis(246, 247, i, 0.05)
    print()


# For each event, we failed to reject the null hypothesis.
# 
# In this case, this is the result we want. The results confirm that there is no statistically significant difference between our control groups, meaning the that the groups were split properly.

# **Do the same thing for the group with altered fonts. Compare the results with those of each of the control groups for each event in isolation. Compare the results with the combined results for the control groups. What conclusions can we draw from the experiment?**

# The hypothesis we formulated before apply here as well.

# - Comparison of each of the control groups (246, 247) with the test group (248) for each event in isolation:

# 1. Compare the groups 246 and 248:

# In[41]:


# loop through all the events and apply the test function
for i in pivot['event_name'].unique():
    check_hypothesis(246, 248, i, 0.05)
    print()


# 2. Compare the groups 247 and 248:

# In[42]:


# loop through all the events and apply the test function
for i in pivot['event_name'].unique():
    check_hypothesis(247, 248, i, 0.05)
    print()


# **Conclusion:**
# 
# For both comparisons, 246 & 248, and 247 & 248 we failed to reject the null hypothesis.
# This means that there is no statistically significant difference between the proportions of the control groups and the test group.

# - Comparison of combined control groups (246 + 247) with the test group (248) for each event:

# First, we need to combine the control groups.
# We'll simply alter the pivot table we created earlier:

# In[43]:


# add the values in 246 and 247 together to group 250 which is the combination of both control groups
pivot[250] = pivot[246] + pivot[247]
pivot


# Now, compare the groups 250 and 248: 

# In[44]:


# loop through all the events and apply the test function
for i in pivot.event_name.unique():
    check_hypothesis(250, 248, i, 0.05)
    print()


# **Conclusion:**
# 
# Same result as with the isolation tests, we failed to reject the null hypothesis. Meaning there is no statistically significance difference between the control groups combined (246 + 247 = 250) and the test group (248).

# **What significance level have we set to test the statistical hypotheses mentioned above? Calculate how many statistical hypothesis tests we carried out. With a statistical significance level of 0.1, one in 10 results could be false. What should the significance level be?**

# We set a significance level of 5% for each of the tests conducted earlier.
# 
# We compared 4 pairs, and for each pair we conducted 4 tests. So, overall we carried out 16 statistical hypothesis tests.
# Since we failed to reject all 16 tests, it's impossible that we made a type I error (Reject a true null hypotesis).
# 
# So, there is no reason for us to change the significance level and run the tests again.

# ## Conclusion

# To sum up, at this stage we studied the results of the experiment.
# 
# Initially, we conducted proportions test for each event in the control groups (246 & 247). We searched for statistically significant differences between samples 246 and 247.
# 
# - The null hypothesis was: For group A and group B, the shares of users who reached a certain event out of the total users at each group are the same.
# 
# 
# - The alternative hypothesis is: For group A and group B, the shares of users who reached a certain event out of the total users at each group are not the same
# 
# We used a significance criterion (alpha) of 5% since it's the golden standard for the industry.
# 
# We failed to reject the null hypothesis for each of the four events, meaning that the control groups were split properly.
# 
# Later, we did the same for the test group. We compared the results with those of each of the control groups for each event in isolation, and compared the results with the combined results for the control groups.
# 
# We failed to reject the null hypothesis in all of the tests. This means that there is no statistically significant difference between the proportions of the control groups and the test group (both in isolation and combined).
# 
# All of the above means that, the test group didn't perform statistically better than both control groups.
# 
# Changing the fonts didn't improve any of the conversion rates.

# # Overall conclusion

# <a class="anchor" id="conclusion"></a>
# [Go back to the Table of Contents](#table_of_contents)

# In conclusion, we loaded an event tracking data file to the variable `data`.
# 
# From the general look at the data, we found the following issues:
# - Columns names are not clear and contain uppercase letters, which is a bad practice.
# 
# 
# - Values in `EventTimestamp` have the wrong format and data type.
# 
# Afterwards, we addressed the issues we mentioned earlier:
# 1. We renamed the columns, made the names clearer, and replaced uppercase letters with lowercase letters, as expected from a python programmer.
# 
# 
# 2. We converted the `event_timestamp` column's values to a date and time format. Moreover, We added a `date` column to display the dates of the logs without time indicators.
# 
# We didn't find any missing values in `data`.
# 
# Next, we used the `describe()` method to get basic statistical information about the data, here are general interest points:
# 
# - The test duration was from 2019-07-25 to 2019-08-08.
# 
# 
# - 'MainScreenAppear' is the most frequent event in `data` with 119,101 instances.
# 
# 
# - '2019-08-06' is the most frequent date in `data` with 36,270 instances.
# 
# Because we found only five unique values in `event_name`. We changed the column's data type to `category` to save memory space.
# 
# Eventually, we searched for duplicates in `data`, we found 413 duplicates.
# These duplicates accounted for less than 1% of the data, so deleting them didn't impact our analysis significantly.
# 
# Later, we studied the data, here are the main points of interest we were able to draw:
# 
# - The number of events in the logs is 243,713.
# 
# 
# - The average number of events per user is 32.28.
# 
# 
# - The test duration was from 2019-07-25 to 2019-08-08, but data before August 1 was incomplete. Therefore, we filtered the data and used only data from August 1, 2019.
# 
# 
# - The number of events lost after the filtering is 1989, which accounted for 0.82% of the original data.
# 
# 
# 
# - The number of users lost after the filtering is 13, which accounted for 0.17% of the original data.
# 
# 
# - We found 2,484 unique users for group 246 (first control group), 2,517 unique users for 247 (second control group), and 2,537 unique users for group 248 (test group).
# 
# Next, we studied the event funnel, we found the following points of interest:
# 
# - We found five different events are in the logs and their frequency of occurrence. Also, we found the number of users who performed each action and their share out of total users. Here they are sorted in descending order:
# 
#     1. 'MainScreenAppear', 7,423 users, 98.45% share.
#     2. 'OfferScreenAppear', 4,597 users, 60.98% share.
#     3. 'CartScreenAppear', 3,736 users, 49.56% share.
#     4. 'PaymentScreenSuccessful', 3,540 users, 46.96% share.
#     5. 'Tutorial', 843 users, 11.18% share.
# 
# 
# - Afterwards, we excluded 'Tutorial' event from the rest of project, because it's irrelevant for the event funnel.
# 
# 
# - Not all sequences of actions contain all of the actions, but the most 'popular' sequence of actions (with all the actions taken) with 912 instances is:
# 
#     1. MainScreenAppear
#     2. OffersScreenAppear
#     3. CartScreenAppear
#     4. PaymentScreenSuccessful
# 
# 
# - For all experiment groups, the largest chunk of users are lost at the 'OffersScreenAppear', almost 40% of users in all groups are lost by the time they reach this event.
# 
# 
# 
# -  Share of users make the entire journey from their first event to payment:
#     - For group number 246: 49% of users reach 'PaymentScreenSuccessful' event.
# 
#     - For group number 247: 47% of users reach 'PaymentScreenSuccessful' event.
# 
#     - For group number 248: 47% of users reach 'PaymentScreenSuccessful' event.
#     
# Making the offers in 'OffersScreenAppear' more attractive and irresistible to users could lead to a higher conversion rate at this stage, therefore leading to higher conversion to purchases.
#     
# Eventually, we studied the results of the experiment.
# 
# Initially, we conducted proportions test for each event in the control groups (246 & 247). We searched for statistically significant difference between samples 246 and 247.
# 
# - The null hypothesis was: For group A and group B, the shares of users who reached a certain event out of the total users at each group are the same.
# 
# 
# - The alternative hypothesis is: For group A and group B, the shares of users who reached a certain event out of the total users at each group are not the same
# 
# We used significance criterion (alpha) of 5% since it's the golden standard for the industry.
# 
# We failed to reject the null hypothesis for each of the four events, meaning the that the control groups were split properly.
# 
# Later, we did the same for the test group. We compared the results with those of each of the control groups for each event in isolation, and compared the results with the combined results for the control groups.
# 
# We failed to reject the null hypothesis in all of the tests. This means that there is no statistically significant difference between the proportions of the control groups and the test group (both in isolation and combined).
# 
# All of the above means that, the test group didn't perform statistically better than both control groups.
# 
# Changing the fonts didn't improve any of the conversion rates.
