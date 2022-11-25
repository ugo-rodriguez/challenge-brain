# path = "/NIRAL/work/ugor/source/challenge-brain/Pytorch-lightning/challenge-brain-pytorch-lightning/Data/dataV06_V12_test.txt"
# list_information = open(path,"r").read().splitlines()

# with open("dataV06_V12_test2.txt", 'w') as f:
#     for patient in list_information:
#         f.write(patient + ',0,left' + '\n')
#         f.write(patient + ',0,right' + '\n')
#         f.write(patient + ',1,left' + '\n')
#         f.write(patient + ',1,right' + '\n')
        

import csv

with open('dataV06_V12_val2.csv','w',newline='') as fichiercsv:
    writer=csv.writer(fichiercsv)
    writer.writerow(['Subject_ID', 'Age', 'E-mail'])


print(2)