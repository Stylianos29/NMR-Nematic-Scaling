import pandas as pd
import re
import pprint

#Passing the content of the .csv file to a Pandas DataFrame
df = pd.read_csv(r'D:\Dropbox (Personal)\Purdue University\2020_C (Fall Term)\PHYS590 (NMR)\Python Programs\NMR-Nematic-Scaling\NMR_measurements\Raw Recovery Data x=0.05898 11.7T.csv')

#Exctracting the column names of the DataFrame
list_of_columns = list(df.columns)

#Exctracting the temperature values from the column names of the DataFrame in an ordered list
list_of_Temperatures = []
for column_label in list_of_columns:
    pattern = "Co5898_(.*?)K"
    substring = re.search(pattern, column_label).group(1)
    substring = substring.replace('p','.')
    list_of_Temperatures.append(float(substring))
#Create a sorted list of the temperature values
list_of_Temperatures = sorted(set(list_of_Temperatures))

#Finding the pairs of columns correspoding to a certain temperature value
Magnetization_recovery_dict = {}
for temperature in list_of_Temperatures:
    temperature_string = '_'+str(temperature)+'K'
    temperature_string = temperature_string.replace('.','p')
    temperature_string = temperature_string.replace('p0','')
    names_pattern = ['Recovery time', 'Recovery Magnetization']
    for column_label in list_of_columns:
        if temperature_string in column_label:
            if 'twait_' in column_label:
                names_pattern[0] = column_label
            else:
                names_pattern[1] = column_label
    Magnetization_recovery_dict[temperature] = names_pattern

#Printing the Magnetization recovery dictionary neatly
pprint.pprint(Magnetization_recovery_dict)

#Exporting the content of the Magnetization_recovery_dict dictionary to a text file 
with open('NMR-Nematic-Scaling\Output_Files\Text_files\Magnetization_recovery_pairs_dictionary.txt','w') as data:
	data.write(str(Magnetization_recovery_dict))

#Presening data in pairs for each temperature value
# for temperature_value in list_of_Temperatures:
#     df[Magnetization_recovery_dict[temperature_value]]
