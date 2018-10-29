# Signature_Detection

Prerequisite: 
Tensorflow installed.
Anaconda installed.
CV2 installed.
Numpy installed.
Data set.

Process of working:
1. You will collect signatures from persons, let say you have collected signatures from two person AAA, BBB.


2. Create a folder named AAA and BBB. 

3. Then put person AAA signatures to AAA folder and BBB signatures to BBB folder.

4. Then cut 20% of signatures photo from AAA and paste it in test_own_data folder 
randomly and also same amount from BBB.

5. Change the epoch size in last line of SystemTraining.py with your data suitability.

6. Then Run the file SystemTraining.py and leave it for enough time to run. It will generate some files of learning.

7. predictAll.py will predict all the signatures from test_own_data folder.
