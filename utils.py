import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats 

from sklearn.preprocessing import StandardScaler, normalize

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance



# 1. Year of study
def year_study(time):
    if time == 'Graduate student':
        return 5
    elif time == 'First year':
        return 1
    elif time == 'Second year':
        return 2
    elif time == 'Third year':
        return 3
    elif time == 'Fourth year':
        return 4
    else:   
        raise ValueError('Unknown year value')
    
# 2. Gender
def gender(time):
    if time == 'Male':
        return 1
    elif time == 'Female':
        return 2
    else:
        raise ValueError('Unknown gender value')

# 3. Difficulty falling asleep (using frequency categories)
def difficulty(time):
    if time == 'Every night':
        return 1
    elif time == 'Often (5-6 times a week)':
        return 2
    elif time == 'Sometimes (3-4 times a week)':
        return 3
    elif time == 'Rarely (1-2 times a week)':
        return 4
    elif time == 'Never':
        return 5
    else:
        print(time)
        raise ValueError('Unknown difficulty value')

# 4. Hours of sleep on a typical day
def hours_of_sleep(time):
    if time == 'More than 8 hours':
        return 6
    elif time == '7-8 hours':
        return 5
    elif time == '6-7 hours':
        return 4
    elif time == '5-6 hours':
        return 3
    elif time == '4-5 hours':
        return 2
    elif time == 'Less than 4 hours':
        return 1
    else:
        raise ValueError('Unknown hours of sleep value')

# 5. Waking up during the night and trouble falling back asleep
def night_wake(time):
    if time == 'Often (5-6 times a week)':
        return 1
    elif time == 'Sometimes (3-4 times a week)':
        return 2
    elif time == 'Every night':
        return 3
    elif time == 'Rarely (1-2 times a week)':
        return 4
    elif time == 'Never':
        return 5
    else:
        raise ValueError('Unknown night wake value')

# 6. Overall quality of sleep
def sleep_quality(time):
    if time == 'Very poor':
        return 1
    elif time == 'Poor':
        return 2
    elif time == 'Average':
        return 3
    elif time == 'Good':
        return 4
    elif time == 'Very good':
        return 5
    else:
        raise ValueError('Unknown sleep quality value')

# 7. Difficulty concentrating due to lack of sleep
def concentration_difficulty(time):
    if time == 'Never':
        return 5
    elif time == 'Rarely':
        return 4
    elif time == 'Sometimes':
        return 3
    elif time == 'Often':
        return 2
    elif time == 'Always':
        return 1
    else:
        raise ValueError('Unknown concentration difficulty value')

# 8. Fatigue during the day
def fatigue(text):
    # Extract the first word to determine frequency
    freq = text.split()[0]
    mapping = {'Never': 5, 'Rarely': 4, 'Sometimes': 3, 'Often': 2, 'Always': 1}
    if freq in mapping:
        return mapping[freq]
    else:
        raise ValueError('Unknown fatigue frequency value')

# 9. Missing/skipping classes due to sleep issues
def miss_class(text):
    freq = text.split()[0]
    mapping = {'Never': 5, 'Rarely': 4, 'Sometimes': 3, 'Often': 2, 'Always': 1}
    if freq in mapping:
        return mapping[freq]
    else:
        raise ValueError('Unknown miss class frequency value')

# 10. Impact of insufficient sleep on assignments/deadlines
def assignment_impact(text):
    if text == 'No impact':
        return 5
    elif text == 'Minor impact':
        return 4
    elif text == 'Moderate impact':
        return 3
    elif text == 'Major impact':
        return 2
    elif text == 'Severe impact':   
        return 1
    else:
        raise ValueError('Unknown assignment impact value')

# 11. Use of electronic devices before sleep
def device_use(time):
    if time == 'Every night':
        return 1
    elif time == 'Often (5-6 times a week)':
        return 2
    elif time == 'Sometimes (3-4 times a week)':
        return 3
    elif time == 'Rarely (1-2 times a week)':
        return 4
    elif time == 'Never':
        return 5
    else:
        raise ValueError('Unknown device use frequency value')

# 12. Caffeine consumption to stay awake
def caffeine_consumption(time):
    if time == 'Every day':
        return 1
    elif time == 'Often (5-6 times a week)':
        return 2
    elif time == 'Sometimes (3-4 times a week)':
        return 3
    elif time == 'Rarely (1-2 times a week)':
        return 4
    elif time == 'Never':
        return 5
    else:
        raise ValueError('Unknown caffeine consumption value')

# 13. Frequency of physical activity/exercise
def exercise_freq(time):
    if time == 'Every day':
        return 5
    elif time == 'Often (5-6 times a week)':
        return 4
    elif time == 'Sometimes (3-4 times a week)':
        return 3
    elif time == 'Rarely (1-2 times a week)':
        return 2
    elif time == 'Never':
        return 1
    else:
        raise ValueError('Unknown exercise frequency value')

# 14. Stress levels related to academic workload
def stress_levels(text):
    if text == 'No stress':
        return 4
    elif text == 'Low stress':
        return 3
    elif text == 'High stress':
        return 2
    elif text == 'Extremely high stress':
        return 1
    else:
        raise ValueError('Unknown stress level value')

# 15. Overall academic performance
def academic_performance(text):
    if text == 'Poor':
        return 1
    if text == 'Below Average':
        return 2
    elif text == 'Average':
        return 3
    elif text == 'Good':
        return 4
    elif text == 'Excellent':
        return 5
    else:
        raise ValueError('Unknown academic performance value')

def convert_dataset(data):
    # Applying the functions to the DataFrame, converting the correct names
    data['Year_Study'] = data['1. What is your year of study?'].apply(year_study)
    data['Gender'] = data['2. What is your gender?'].apply(gender)
    data['Sleep_Difficulty'] = data['3. How often do you have difficulty falling asleep at night? '].apply(difficulty)
    data['Hours_of_Sleep'] = data['4. On average, how many hours of sleep do you get on a typical day?'].apply(hours_of_sleep)
    data['Night_Wake'] = data['5. How often do you wake up during the night and have trouble falling back asleep?'].apply(night_wake)
    data['Sleep_Quality'] = data['6. How would you rate the overall quality of your sleep?'].apply(sleep_quality)
    data['Concentration_Difficulty'] = data['7. How often do you experience difficulty concentrating during lectures or studying due to lack of sleep?'].apply(concentration_difficulty)
    data['Fatigue'] = data['8. How often do you feel fatigued during the day, affecting your ability to study or attend classes?'].apply(fatigue)
    data['Missed_Classes'] = data['9. How often do you miss or skip classes due to sleep-related issues (e.g., insomnia, feeling tired)?'].apply(miss_class)
    data['Assignment_Impact'] = data['10. How would you describe the impact of insufficient sleep on your ability to complete assignments and meet deadlines?'].apply(assignment_impact)
    data['Device_Use'] = data['11. How often do you use electronic devices (e.g., phone, computer) before going to sleep?'].apply(device_use)
    data['Caffeine_Consumption'] = data['12. How often do you consume caffeine (coffee, energy drinks) to stay awake or alert?'].apply(caffeine_consumption)
    data['Exercise_Frequency'] = data['13. How often do you engage in physical activity or exercise?'].apply(exercise_freq)
    data['Stress_Levels'] = data['14. How would you describe your stress levels related to academic workload?'].apply(stress_levels)
    data['Academic_Performance'] = data['15. How would you rate your overall academic performance (GPA or grades) in the past semester?'].apply(academic_performance)
    # Drop the original columns
    data = data.drop(['1. What is your year of study?', '2. What is your gender?',
                    '3. How often do you have difficulty falling asleep at night? ',
                    '4. On average, how many hours of sleep do you get on a typical day?',
                    '5. How often do you wake up during the night and have trouble falling back asleep?',
                    '6. How would you rate the overall quality of your sleep?',
                    '7. How often do you experience difficulty concentrating during lectures or studying due to lack of sleep?',
                    '8. How often do you feel fatigued during the day, affecting your ability to study or attend classes?',
                    '9. How often do you miss or skip classes due to sleep-related issues (e.g., insomnia, feeling tired)?',
                    '10. How would you describe the impact of insufficient sleep on your ability to complete assignments and meet deadlines?',
                    '11. How often do you use electronic devices (e.g., phone, computer) before going to sleep?',
                    '12. How often do you consume caffeine (coffee, energy drinks) to stay awake or alert?',
                    '13. How often do you engage in physical activity or exercise?',
                    '14. How would you describe your stress levels related to academic workload?',
                    '15. How would you rate your overall academic performance (GPA or grades) in the past semester?'], axis=1)
    return data


def plot_correlations(data):
    plt.figure(figsize=(20, 10))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')

def discrete_analysis(discrete_feat, data, features='Hours_of_Sleep'):
    for feat in discrete_feat:
        # import pdb; pdb.set_trace()
        groups = data.groupby(feat)[f'{features}'].apply(list)
        if len(groups)>1:
            _, p_value = stats.kruskal(*groups)
        else:
            p_value = 0.0
        
        plt.figure(figsize=(12,4))
        plt.subplot(121)
        crosstab = data.groupby([f'{features}',feat]).size().unstack()
        crosstab.plot(kind='bar',stacked=True,cmap='rocket',ax=plt.gca())
        plt.box(False)

        plt.subplot(122)
        sns.lineplot(data=data.groupby(feat)[f'{features}'].mean(),color='tomato',linewidth=3,marker='o',markersize=10)
        plt.box(False)
        
        plt.suptitle(f'Sleep Quality by {feat} [P Value :{p_value:0.2f}]')
        plt.tight_layout()
        plt.show()


def importance_analysis(data, features='Academic_Performance'):
    
    X, y = data.drop(f'{features}',axis=1),data[f'{features}']


    model = RandomForestRegressor(n_estimators=1000,random_state=123)
    model.fit(X,y)

    results = permutation_importance(model, X, y,random_state=123)

    imp_df = pd.DataFrame({
        'Importance': results.importances_mean
    },index=X.columns).sort_values(by='Importance',ascending=False)
    sns.barplot(data=imp_df,x='Importance',y=imp_df.index,palette='viridis')